"""Measure EEG POD in fresh processes (Linux/macOS), without using a GPU."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import psutil
import xarray as xr

from demos.eeg_analysis import filtered_eeg_file
from demos.eeg_utils import alljoined_paths, recording_manifest
from svdrom import POD


def run_case(directory: Path, count: int, chunk_samples: int, workers: int) -> dict:
    """Fit a recording prefix and score the full held-out file with eight modes."""
    paths = alljoined_paths(directory)
    if not 1 <= count < len(paths) or chunk_samples <= 0 or workers <= 0:
        msg = "Use 1-12 training files and positive chunk/worker counts."
        raise ValueError(msg)
    manifest = recording_manifest(paths[:count] + [paths[-1]])
    channels = tuple(manifest[0]["channels"])
    started = time.perf_counter()
    with dask.config.set(scheduler="threads", num_workers=workers):
        arrays = []
        for path, item in zip(paths[:count] + [paths[-1]], manifest, strict=True):
            filtered = filtered_eeg_file(
                path, item["n_samples"], channels, chunk_samples, item["sfreq"]
            )
            mean = filtered.astype(np.float64).mean(axis=1).compute()
            arrays.append(
                (filtered.astype(np.float64) - mean[:, None]).astype(np.float32)
            )
        conditioning_seconds = time.perf_counter() - started
        train = xr.DataArray(
            da.concatenate(arrays[:-1], axis=1), dims=("channel", "sample")
        )
        test = xr.DataArray(arrays[-1], dims=("channel", "sample"))
        n_train = train.sizes["sample"]
        started = time.perf_counter()
        model = POD(
            n_modes=8,
            svd_algorithm="covariance",
            time_dimension="sample",
            compute_time_coeffs=False,
            compute_energy_ratio=True,
            remove_mean=False,
        ).fit(train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        coefficients = model.transform(test, compute=False)
        reconstruction = (model.modes @ coefficients) * np.sqrt(n_train)
        baseline, error = dask.compute(
            (test.astype(np.float64) ** 2).sum(),
            ((test - reconstruction).astype(np.float64) ** 2).sum(),
        )
        evaluation_seconds = time.perf_counter() - started
    return {
        "training_files": count,
        "training_samples": n_train,
        "training_gib": train.nbytes / 1024**3,
        "chunk_samples": chunk_samples,
        "workers": workers,
        "conditioning_seconds": conditioning_seconds,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
        "held_out_energy": 1 - float(error) / float(baseline),
        "training_energy": float(np.sum(model.explained_energy_ratio)),
        "manifest": manifest,
    }


def main() -> None:
    """Run each requested size in a fresh process and save benchmark metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SVDROM_EEG_DIR", "demos/data/alljoined1")),
    )
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 4, 12])
    parser.add_argument("--chunk-samples", type=int, default=131_072)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("demos/eeg_benchmark.json"))
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        import resource

        with contextlib.redirect_stdout(sys.stderr):
            result = run_case(
                args.data_dir, args.worker, args.chunk_samples, args.workers
            )
        # ru_maxrss includes imports, fitting and evaluation, but not the parent.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_rss_gib"] = peak / (
            1024**3 if sys.platform == "darwin" else 1024**2
        )
        print(json.dumps(result))
        return
    if sys.platform not in ("linux", "darwin"):
        parser.error("Peak-RSS measurement currently supports Linux and macOS.")
    if any(not 1 <= count <= 12 for count in args.counts):
        parser.error("--counts must contain values from 1 to 12.")
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        environment[name] = "1"
    results = []
    available = psutil.virtual_memory()
    for count in args.counts:
        print(f"Benchmarking {count} training files...", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "demos.benchmark_eeg",
                "--worker",
                str(count),
                "--data-dir",
                str(args.data_dir.resolve()),
                "--chunk-samples",
                str(args.chunk_samples),
                "--workers",
                str(args.workers),
            ],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        results.append(json.loads(completed.stdout))
        print(
            json.dumps({k: v for k, v in results[-1].items() if k != "manifest"}),
            flush=True,
        )
    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "total_ram_gib": available.total / 1024**3,
        "available_ram_at_start_gib": available.available / 1024**3,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("svdrom", "numpy", "dask", "xarray", "mne", "scipy", "psutil")
        },
        "blas_threads": 1,
        "gpu_used": False,
        "preprocessing": {
            "high_pass_hz": 0.5,
            "halo_seconds": 32,
            "filter_order": 4,
            "zero_phase": True,
            "per_recording_center": True,
        },
        "measurement": "Fresh process peak RSS, including imports, fit and evaluation.",
        "timing_context": "Shared host; filesystem cache is not cleared between runs.",
        "cases": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
