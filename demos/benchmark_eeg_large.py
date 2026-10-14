"""Benchmark the complete Alljoined-1.6M corpus and verify disk encodings."""

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
import h5py
import numpy as np
import psutil

from demos.eeg_analysis import (
    band_power_errors,
    covariance_scores,
    participant_validation,
)
from demos.eeg_large import (
    CHANNELS,
    PREPROCESSING,
    cache_manifest,
    encode_cached,
    encoded_spectra,
    fit_cached,
    prepare_cache,
    save_encoded_spectra,
    verify_encoding,
)


def run_case(cache: Path, count: int, chunk_samples: int, workers: int) -> dict:
    """Fit the requested participant prefix with lazy coefficients and 24 modes."""
    started = time.perf_counter()
    with dask.config.set(scheduler="threads", num_workers=workers):
        model = fit_cached(cache, count, chunk_samples)
    fit_seconds = time.perf_counter() - started
    manifest = cache_manifest(cache)
    test_covariance = np.zeros((len(CHANNELS), len(CHANNELS)))
    n_test = 0
    with h5py.File(cache, "r") as store:
        for item in manifest["records"]:
            if item["subject"] == 20:
                test_covariance += (
                    store[f"{item['key']}/covariance"][:] * item["n_samples"]
                )
                n_test += item["n_samples"]
    scores = covariance_scores(
        test_covariance / n_test, model.modes.values, (4, 8, 16, 24)
    )
    return {
        "training_participants": count,
        "training_recordings": sum(i["subject"] <= count for i in manifest["records"]),
        "training_samples": model.time_coeffs.sizes["sample"],
        "training_gib": model.time_coeffs.sizes["sample"] * len(CHANNELS) * 4 / 2**30,
        "fit_seconds": fit_seconds,
        "training_energy": float(np.sum(model.explained_energy_ratio)),
        "held_out_energy": {
            str(rank): score["energy"] for rank, score in scores.items()
        },
        "chunk_samples": chunk_samples,
        "workers": workers,
    }


def run_encoding(cache: Path, output: Path, chunk_samples: int, workers: int) -> dict:
    """Fit all training people, encode the full corpus at two ranks, and decode it."""
    started = time.perf_counter()
    with dask.config.set(scheduler="threads", num_workers=workers):
        model = fit_cached(cache, 19, chunk_samples)
    fit_seconds = time.perf_counter() - started
    results = []
    for rank in (8, 16):
        path = output / f"rank{rank:02d}.h5"
        started = time.perf_counter()
        result = encode_cached(cache, model, path, rank, chunk_samples)
        result["encoding_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        result["verification"] = verify_encoding(cache, path, chunk_samples)
        result["verification_seconds"] = time.perf_counter() - started
        result["nominal_compression"] = result["matrix_bytes"] / result["payload_bytes"]
        result["file_compression"] = result["matrix_bytes"] / result["file_bytes"]
        results.append(result)
    manifest = cache_manifest(cache)
    stats = []
    with h5py.File(cache, "r") as store:
        for item in manifest["records"]:
            stats.append(
                {
                    "n_samples": item["n_samples"],
                    "covariance": store[f"{item['key']}/covariance"][:],
                }
            )
    folds = participant_validation(
        stats, [i["subject"] for i in manifest["records"]], (4, 8, 16, 24)
    )
    started = time.perf_counter()
    with dask.config.set(scheduler="threads", num_workers=workers):
        spectra = encoded_spectra(cache, tuple(Path(r["file"]) for r in results))
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "30-45 Hz": (30, 45),
        "50 Hz noise": (49, 51),
        "60 Hz noise": (59, 61),
    }
    errors = band_power_errors(spectra["frequencies"], spectra["psd"], bands)
    spectral_path = output / "spectra.h5"
    save_encoded_spectra(spectral_path, spectra)
    # Scores contain only Python scalars, unlike covariance_scores' extra arrays.
    return {
        "fit_seconds": fit_seconds,
        "encodings": results,
        "participant_folds": folds,
        "spectra": {
            "file": str(spectral_path),
            "subject": 20,
            "n_windows": spectra["n_windows"],
            "seconds": time.perf_counter() - started,
            "median_absolute_band_power_error_percent": {
                name: (100 * np.nanmedian(np.abs(values), axis=1)).tolist()
                for name, values in errors.items()
            },
        },
    }


def main() -> None:
    """Prepare once; launch sequential fresh workers for scaling and round trips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            os.environ.get("SVDROM_EEG_LARGE_DIR", "demos/data/alljoined16/raw_eeg")
        ),
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("demos/data/alljoined16/conditioned.h5")
    )
    parser.add_argument(
        "--encoded-dir", type=Path, default=Path("demos/data/alljoined16/encoded")
    )
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 5, 10, 19])
    parser.add_argument("--chunk-samples", type=int, default=16_384)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path("demos/eeg_large_benchmark.json")
    )
    parser.add_argument(
        "--preparation-report",
        type=Path,
        help="Reuse the JSON result of a previously measured preparation worker.",
    )
    parser.add_argument(
        "--worker", choices=["prepare", "fit", "encode"], help=argparse.SUPPRESS
    )
    parser.add_argument("--count", type=int, default=19, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform not in ("linux", "darwin"):
        parser.error("Peak RSS measurement supports Linux and macOS.")
    if (
        any(not 1 <= count <= 19 for count in args.counts)
        or args.chunk_samples <= 0
        or args.workers <= 0
    ):
        parser.error("Use 1-19 training participants and positive chunk/worker counts.")
    if args.worker:
        import resource

        with contextlib.redirect_stdout(sys.stderr):
            if args.worker == "prepare":
                reused = 0
                if args.cache.exists():
                    with h5py.File(args.cache, "r") as store:
                        reused = sum(
                            group.attrs.get("complete", False)
                            for group in store.values()
                        )
                started = time.perf_counter()
                manifest = prepare_cache(args.data_dir, args.cache, args.chunk_samples)
                result = {
                    "manifest": manifest,
                    "seconds": time.perf_counter() - started,
                    "cache_bytes": args.cache.stat().st_size,
                    "recordings_reused": int(reused),
                    "recordings_prepared": len(manifest["records"]) - int(reused),
                }
            elif args.worker == "fit":
                result = run_case(
                    args.cache, args.count, args.chunk_samples, args.workers
                )
            else:
                result = run_encoding(
                    args.cache, args.encoded_dir, args.chunk_samples, args.workers
                )
        result["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            2**30 if sys.platform == "darwin" else 2**20
        )
        print(json.dumps(result))
        return

    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        environment[name] = "1"
    available = psutil.virtual_memory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "total_ram_gib": available.total / 2**30,
        "available_ram_at_start_gib": available.available / 2**30,
        "gpu_used": False,
        "blas_threads": 1,
        "preprocessing": PREPROCESSING,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "svdrom",
                "numpy",
                "dask",
                "xarray",
                "mne",
                "scipy",
                "h5py",
                "psutil",
            )
        },
        "measurement": (
            "Fresh worker peak RSS; excludes other processes and filesystem cache."
        ),
        "timing_context": (
            "Shared host; cache not cleared. Fit uses prepared HDF5, "
            "preparation measured separately."
        ),
        "cases": [],
    }
    stages = (
        [("prepare", 19)] + [("fit", count) for count in args.counts] + [("encode", 19)]
    )
    if args.preparation_report:
        preparation = json.loads(args.preparation_report.read_text())
        if (
            preparation["manifest"] != cache_manifest(args.cache)
            or preparation["cache_bytes"] != args.cache.stat().st_size
        ):
            msg = "Preparation report does not describe this complete cache."
            raise ValueError(msg)
        report["prepare"] = preparation
        stages = stages[1:]
    for stage, count in stages:
        print(f"Stage {stage}, training participants {count}", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "demos.benchmark_eeg_large",
                "--worker",
                stage,
                "--count",
                str(count),
                "--data-dir",
                str(args.data_dir.resolve()),
                "--cache",
                str(args.cache.resolve()),
                "--encoded-dir",
                str(args.encoded_dir.resolve()),
                "--chunk-samples",
                str(args.chunk_samples),
                "--workers",
                str(args.workers),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            msg = f"{stage} failed: {completed.stderr}"
            raise RuntimeError(msg)
        result = json.loads(completed.stdout)
        if stage == "fit":
            report["cases"].append(result)
        else:
            report[stage] = result
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"Finished {stage}: peak RSS {result['peak_rss_gib']:.2f} GiB", flush=True
        )


if __name__ == "__main__":
    main()
