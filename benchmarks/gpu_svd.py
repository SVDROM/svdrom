"""Benchmark disk-backed weather arrays with Dask's NumPy and CuPy backends.

Run ``uv run python benchmarks/gpu_svd.py --help`` for usage. This script does
not import svdrom unless a ``--model svdrom-*`` mode is selected. It uses one worker
so per-block transfer timings are additive.
Disk-backed runs may benefit from the operating system's filesystem cache.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import dask
import dask.array as da
import numpy as np
import psutil
import xarray as xr
from dask.callbacks import Callback
from dask.utils import parse_bytes


class Transfers:
    """Measure synchronized transfers inside a single-process Dask scheduler."""

    def __init__(self, xp: ModuleType, sync_before: bool = True) -> None:
        self.xp = xp
        self.sync_before = sync_before
        self.upload_seconds = 0.0
        self.download_seconds = 0.0

    def upload(self, block: Any) -> Any:
        """Copy one input block and record its host-to-device elapsed time."""
        if self.sync_before:
            self.xp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        result = self.xp.asarray(block)
        self.xp.cuda.runtime.deviceSynchronize()
        self.upload_seconds += time.perf_counter() - start
        return result

    def download(self, block: Any) -> np.ndarray:
        """Copy one output block and record its device-to-host elapsed time."""
        self.xp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        result = self.xp.asnumpy(block)
        self.download_seconds += time.perf_counter() - start
        return result


class DeviceMemory:
    """Sample total GPU memory use, including CUDA workspace, every 10 ms."""

    def __init__(self, xp: ModuleType | None, host_limit: int | None = None) -> None:
        self.xp = xp
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._device = 0
        self.host_limit = host_limit
        self.peak_host_rss = 0
        self.minimum_host_available = psutil.virtual_memory().available
        self._process = psutil.Process()
        self._callback = Callback(pretask=self.check_budget)

    def start(self) -> None:
        """Start sampling on the calling thread's CUDA device."""
        self.peak_host_rss = self._process.memory_info().rss
        if self.xp is not None:
            self._device = self.xp.cuda.runtime.getDevice()
            free, total = self.xp.cuda.runtime.memGetInfo()
            self.baseline = self.peak = total - free
        if self.host_limit is not None:
            self._callback.register()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        if self.xp is not None:
            self.xp.cuda.Device(self._device).use()
        while not self._stop.is_set():
            self.peak_host_rss = max(
                self.peak_host_rss, self._process.memory_info().rss
            )
            self.minimum_host_available = min(
                self.minimum_host_available, psutil.virtual_memory().available
            )
            if self.xp is not None:
                free, total = self.xp.cuda.runtime.memGetInfo()
                self.peak = max(self.peak, total - free)
            self._stop.wait(0.01)

    def check_budget(self, *_args: Any) -> None:
        """Abort between Dask tasks before host memory approaches WSL's limit."""
        if (
            self.host_limit is not None
            and self._process.memory_info().rss > self.host_limit
        ):
            msg = "The benchmark exceeded its host-memory budget; aborting before OOM."
            raise MemoryError(msg)

    def stop(self) -> None:
        """Stop sampling and wait for the sampler to exit."""
        self._stop.set()
        if self.host_limit is not None:
            self._callback.unregister()
        if self._thread is not None:
            self._thread.join()


def load_matrix(args: argparse.Namespace) -> da.Array:
    """Open a weather variable lazily and stack its non-time dimensions."""
    manifest_path = Path(args.path) / "acquisition.json"
    if (
        manifest_path.is_file()
        and not json.loads(manifest_path.read_text())["complete"]
    ):
        msg = "The dataset acquisition is incomplete; finish preparing it first."
        raise ValueError(msg)
    options = {"chunks": "auto"}
    if args.engine:
        options["engine"] = args.engine
    dataset = xr.open_dataset(args.path, **options)
    if args.variable:
        field = dataset[args.variable]
    elif len(dataset.data_vars) == 1:
        field = dataset[next(iter(dataset.data_vars))]
    else:
        msg = "Select a dataset variable with --variable."
        raise ValueError(msg)
    if args.snapshots:
        field = field.isel({args.time_dim: slice(0, args.snapshots)})
    spatial_dims = [dim for dim in field.dims if dim != args.time_dim]
    if not spatial_dims or args.time_dim not in field.dims:
        msg = "Expected a time dimension and at least one spatial dimension."
        raise ValueError(msg)
    field = field.stack(benchmark_samples=spatial_dims).transpose(
        "benchmark_samples", args.time_dim
    )
    return field.data.rechunk((args.spatial_chunk, args.time_chunk))


def decompose(
    values: da.Array,
    algorithm: str,
    rank: int,
    power: int,
    staged: bool = False,
    iterator: str = "power",
) -> tuple[da.Array, da.Array, da.Array]:
    """Build a truncated TSQR or randomized SVD graph with a fixed seed."""
    if algorithm == "tsqr":
        if values.shape[0] >= values.shape[1]:
            values = values.rechunk({1: -1})
        else:
            values = values.rechunk({0: -1})
        u, s, v = da.linalg.svd(values)
        return u[:, :rank], s[:rank], v[:rank, :]
    return da.linalg.svd_compressed(
        values, rank, n_power_iter=power, seed=42, compute=staged, iterator=iterator
    )


def run_case(
    matrix: da.Array,
    backend: str,
    dtype: str,
    residency: str,
    algorithm: str,
    power: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Measure one case, excluding persistence but including final host output."""
    result: dict[str, Any] = {
        "model": args.model,
        "backend": backend,
        "dtype": dtype,
        "residency": residency,
        "algorithm": algorithm,
        "n_power_iter": power,
        "status": "ok",
    }
    result["staged"] = getattr(args, "staged", False)
    result["iterator"] = getattr(args, "iterator", "power")
    values = matrix.astype(dtype)
    host_available = psutil.virtual_memory().available
    if (
        residency == "persisted"
        and backend == "numpy"
        and values.nbytes > host_available * getattr(args, "host_memory_fraction", 0.5)
    ):
        result.update(
            status="skipped", reason="Input exceeds the host persistence budget."
        )
        return result
    xp = importlib.import_module("cupy") if backend == "cupy" else None
    if xp is not None:
        if not xp.is_available():
            msg = "CuPy is installed but no usable CUDA device is available."
            raise RuntimeError(msg)
        xp.get_default_memory_pool().free_all_blocks()
        free, _total = xp.cuda.runtime.memGetInfo()
        if (
            residency == "persisted"
            and values.nbytes > free * args.device_memory_fraction
        ):
            result.update(
                status="skipped",
                reason="Input exceeds persistence budget; use the disk-backed case.",
            )
            return result
    transfers = (
        Transfers(xp, sync_before=getattr(args, "copy_sync", "pre-post") == "pre-post")
        if xp is not None
        else None
    )
    if xp is not None and getattr(args, "device_memory_limit", None):
        xp.get_default_memory_pool().set_limit(
            size=parse_bytes(args.device_memory_limit)
        )
    if transfers is not None and (args.model == "dask-svd" or residency == "persisted"):
        values = values.map_blocks(
            transfers.upload,
            dtype=values.dtype,
            meta=xp.empty((0, 0), dtype=values.dtype),
        )
    persist_seconds = 0.0
    if residency == "persisted":
        start = time.perf_counter()
        values = values.persist()
        if xp is not None:
            xp.cuda.runtime.deviceSynchronize()
        persist_seconds = time.perf_counter() - start

    if args.model != "dask-svd":
        from svdrom import POD, TruncatedSVD

    elapsed = []
    uploads = []
    downloads = []
    peaks = []
    host_peaks = []
    host_available_minima = []
    for repeat in range(args.warmups + args.repeats):
        if transfers is not None:
            transfers.upload_seconds = transfers.download_seconds = 0.0
            xp.cuda.runtime.deviceSynchronize()
        host_limit = getattr(args, "host_memory_limit", None)
        memory = DeviceMemory(xp, parse_bytes(host_limit) if host_limit else None)
        memory.start()
        try:
            start = time.perf_counter()
            if args.model != "dask-svd":
                if args.model == "svdrom-pod":
                    model = POD(
                        args.rank, svd_algorithm=algorithm, compute_energy_ratio=True
                    )
                else:
                    model = TruncatedSVD(args.rank, algorithm=algorithm)
                options = (
                    {
                        "seed": 42,
                        "n_power_iter": power,
                        "compute": result["staged"],
                        "iterator": result["iterator"],
                    }
                    if algorithm == "randomized"
                    else {}
                )
                model.fit(xr.DataArray(values, dims=("space", "time")), **options)
                computed = (
                    model.u.data,
                    model.s,
                    model.time_coeffs.data
                    if args.model == "svdrom-pod"
                    else model.v.data,
                )
            else:
                factors = decompose(
                    values,
                    algorithm,
                    args.rank,
                    power,
                    result["staged"],
                    result["iterator"],
                )
                if transfers is not None:
                    factors = tuple(
                        factor.map_blocks(
                            transfers.download,
                            dtype=factor.dtype,
                            meta=np.empty((0,) * factor.ndim, dtype=factor.dtype),
                        )
                        for factor in factors
                    )
                computed = dask.compute(*factors)
            if xp is not None:
                xp.cuda.runtime.deviceSynchronize()
            seconds = time.perf_counter() - start
        except Exception as exc:
            result.update(
                status="error",
                reason=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.perf_counter() - start,
                peak_host_rss_bytes_sampled=memory.peak_host_rss,
                peak_gpu_bytes_sampled=memory.peak if xp is not None else None,
                minimum_host_available_bytes=memory.minimum_host_available,
            )
            return result
        finally:
            memory.stop()
        if not all(isinstance(factor, np.ndarray) for factor in computed):
            msg = "Expected final SVD/POD factors to be NumPy-backed."
            raise TypeError(msg)
        if not all(np.isfinite(factor).all() for factor in computed):
            msg = "SVD produced non-finite output."
            raise ValueError(msg)
        if repeat >= args.warmups:
            elapsed.append(seconds)
            uploads.append(transfers.upload_seconds if transfers else 0.0)
            downloads.append(transfers.download_seconds if transfers else 0.0)
            peaks.append(memory.peak)
            host_peaks.append(memory.peak_host_rss)
            host_available_minima.append(memory.minimum_host_available)
        print(
            f"{args.model} {backend} {residency} {algorithm} q={power}: "
            f"{'warm-up' if repeat < args.warmups else 'measured'} "
            f"{seconds:.3f} s; host RSS {memory.peak_host_rss / 1024**3:.2f} GiB",
            file=sys.stderr,
            flush=True,
        )
    result.update(
        seconds_median=statistics.median(elapsed),
        seconds_min=min(elapsed),
        seconds_max=max(elapsed),
        seconds_measured=elapsed,
        upload_seconds_median=(
            statistics.median(uploads) if args.model == "dask-svd" else None
        ),
        download_seconds_median=(
            statistics.median(downloads) if args.model == "dask-svd" else None
        ),
        peak_gpu_bytes_sampled=max(peaks) if xp is not None else None,
        persist_seconds=persist_seconds,
        singular_values=computed[1].tolist(),
        peak_host_rss_bytes_sampled=max(host_peaks),
        minimum_host_available_bytes=min(host_available_minima),
        output_backend="numpy",
        retained_factor_bytes=sum(factor.nbytes for factor in computed),
    )
    samples = min(getattr(args, "validation_snapshots", 0), matrix.shape[1])
    if samples:
        indices = np.linspace(0, matrix.shape[1] - 1, samples, dtype=int)
        reference = matrix[:, indices].compute()
        if args.model != "dask-svd":
            predicted = np.stack(
                [model.reconstruct(int(index)).data for index in indices], axis=1
            )
            if args.model == "svdrom-pod":
                result["captured_energy_fraction"] = float(
                    np.sum(model.explained_energy_ratio)
                )
        else:
            predicted = (computed[0] * computed[1]) @ computed[2][:, indices]
        difference = predicted - reference
        result["validation"] = {
            "snapshot_indices": indices.tolist(),
            "rmse": float(np.sqrt(np.mean(difference**2, dtype=np.float64))),
            "relative_l2_error": float(
                np.linalg.norm(difference) / np.linalg.norm(reference)
            ),
            "max_abs_error": float(np.max(np.abs(difference))),
        }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark settings, allowing a CPU-only smoke run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Local Zarr, NetCDF or HDF5 weather dataset")
    parser.add_argument("--variable")
    parser.add_argument("--engine")
    parser.add_argument("--time-dim", default="time")
    parser.add_argument("--snapshots", type=int)
    parser.add_argument("--spatial-chunk", type=int, default=8192)
    parser.add_argument("--time-chunk", type=int, default=256)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument(
        "--model",
        choices=["dask-svd", "svdrom-svd", "svdrom-pod"],
        default="dask-svd",
        help="Raw Dask SVD, public TruncatedSVD, or POD with preprocessing and energy",
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=["tsqr", "randomized-0", "randomized-2"],
        default=["tsqr", "randomized-0", "randomized-2"],
    )
    parser.add_argument(
        "--backend", nargs="+", choices=["numpy", "cupy"], default=["numpy", "cupy"]
    )
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=["float32", "float64"],
        default=["float32", "float64"],
    )
    parser.add_argument(
        "--residency",
        nargs="+",
        choices=["disk", "persisted"],
        default=["disk", "persisted"],
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device-memory-fraction", type=float, default=0.5)
    parser.add_argument("--host-memory-fraction", type=float, default=0.5)
    parser.add_argument("--host-memory-limit", help="Abort budget, e.g. 8 GiB")
    parser.add_argument("--device-memory-limit", help="CuPy pool limit, e.g. 10 GiB")
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Recompute input between randomized SVD passes to bound memory",
    )
    parser.add_argument("--validation-snapshots", type=int, default=0)
    parser.add_argument("--iterator", choices=["power", "QR"], default="power")
    parser.add_argument(
        "--copy-sync",
        choices=["pre-post", "post"],
        default="pre-post",
        help="Raw copy-clock synchronization; post reproduces the early reports",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.rank < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("rank/repeats must be positive and warmups must be nonnegative")
    if not 0 < args.device_memory_fraction < 1:
        parser.error("device-memory-fraction must lie between zero and one")
    if not 0 < args.host_memory_fraction < 1:
        parser.error("host-memory-fraction must lie between zero and one")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and emit machine-readable results plus a Markdown table."""
    args = parse_args(argv)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with dask.config.set({"array.backend": "numpy"}):
        matrix = load_matrix(args)
    if args.rank >= min(matrix.shape):
        msg = "rank must be smaller than both matrix dimensions"
        raise ValueError(msg)
    report: dict[str, Any] = {
        "environment": {
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "dask", "xarray")
            },
            "path": args.path,
            "shape": matrix.shape,
            "chunks": matrix.chunks,
            "rank": args.rank,
            "model": args.model,
            "input_bytes": matrix.nbytes,
            "host_total_bytes": psutil.virtual_memory().total,
            "host_available_bytes": psutil.virtual_memory().available,
            "staged": args.staged,
            "iterator": args.iterator,
            "copy_sync": args.copy_sync,
            "host_memory_limit": args.host_memory_limit,
            "device_memory_limit": args.device_memory_limit,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "scheduler": "threads, one worker",
            "blas_threads": {
                key: os.environ.get(key)
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
            },
            "disk_cache": "OS filesystem cache is not cleared",
            "transfer_timing": (
                (
                    "Device synchronized before and after each per-block copy"
                    if args.copy_sync == "pre-post"
                    else "Device synchronized after each upload; copy-window estimate"
                )
                if args.model == "dask-svd"
                else "Internal library transfers included in wall time; not profiled"
            ),
            "input_conversion": (
                "Instrumented map_blocks(cupy.asarray)"
                if args.model == "dask-svd"
                else "Host-backed disk input is converted by the public library API"
            ),
        },
        "results": [],
    }
    print("| Backend | Dtype | Residency | SVD | Power | Seconds | Status |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for backend in args.backend:
        with dask.config.set(
            {"array.backend": backend, "scheduler": "threads", "num_workers": 1}
        ):
            for dtype in args.dtype:
                for residency in args.residency:
                    for setting in args.algorithm:
                        algorithm = "tsqr" if setting == "tsqr" else "randomized"
                        power = 2 if setting == "randomized-2" else 0
                        try:
                            row = run_case(
                                matrix,
                                backend,
                                dtype,
                                residency,
                                algorithm,
                                power,
                                args,
                            )
                        except Exception as exc:
                            row = {
                                "backend": backend,
                                "dtype": dtype,
                                "residency": residency,
                                "algorithm": algorithm,
                                "n_power_iter": power,
                                "status": "error",
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                        report["results"].append(row)
                        if row["status"] == "error":
                            print(row["reason"], file=sys.stderr, flush=True)
                        if backend == "cupy" and row["status"] == "ok":
                            cp = importlib.import_module("cupy")
                            report["environment"]["gpu"] = {
                                "name": cp.cuda.runtime.getDeviceProperties(0)[
                                    "name"
                                ].decode(),
                                "driver_version": cp.cuda.runtime.driverGetVersion(),
                                "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
                                "cupy_version": cp.__version__,
                            }
                        seconds = (
                            f"{row['seconds_median']:.4f}"
                            if row["status"] == "ok"
                            else "—"
                        )
                        print(
                            f"| {backend} | {dtype} | {residency} | {algorithm} | "
                            f"{power} | {seconds} | {row['status']} |",
                            flush=True,
                        )
                        if args.output:
                            args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(any(row["status"] == "error" for row in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
