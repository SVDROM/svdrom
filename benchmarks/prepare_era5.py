"""Download one native-precision ERA5 surface field with resumable chunk copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numcodecs
import numpy as np
import xarray as xr
import zarr

SOURCE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
GCS_SOURCE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def fetch(url: str, attempts: int = 5) -> bytes:
    """Read a metadata file or one encoded chunk, retrying transient failures."""
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=60) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt, 8))
    msg = "No download attempts were made."
    raise ValueError(msg)


def validate_chunk(payload: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
    """Check a native chunk's decoded size and finite values without changing dtype."""
    decoded = numcodecs.get_codec(metadata["compressor"]).decode(payload)
    values = np.frombuffer(decoded, dtype=metadata["dtype"])
    expected = int(np.prod(metadata["chunks"]))
    if values.size != expected or not np.isfinite(values).all():
        msg = "An ERA5 chunk has the wrong size or contains missing/non-finite data."
        raise ValueError(msg)
    return {
        "encoded_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def copy_chunk(
    target: Path,
    variable: str,
    source_index: int,
    local_index: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Copy and verify one complete encoded source chunk with atomic replacement."""
    destination = target / variable / f"{local_index}.0.0"
    if destination.exists():
        payload = destination.read_bytes()
        try:
            record = validate_chunk(payload, metadata)
        except (ValueError, RuntimeError):
            destination.unlink()
        else:
            return dict(
                record, local_index=local_index, source_index=source_index, reused=True
            )
    payload = fetch(f"{SOURCE}/{variable}/{source_index}.0.0")
    record = validate_chunk(payload, metadata)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_bytes(payload)
    partial.replace(destination)
    return dict(
        record, local_index=local_index, source_index=source_index, reused=False
    )


def initialize(target: Path, variable: str, year: int) -> tuple[dict[str, Any], int]:
    """Create coordinates and Zarr metadata, or validate a resumed store."""
    metadata = json.loads(fetch(f"{SOURCE}/{variable}/.zarray"))
    attributes = json.loads(fetch(f"{SOURCE}/{variable}/.zattrs"))
    dims = attributes["_ARRAY_DIMENSIONS"]
    if dims != ["time", "latitude", "longitude"] or metadata["chunks"] != [
        1,
        721,
        1440,
    ]:
        msg = "This downloader supports the source's hourly full-grid surface chunks."
        raise ValueError(msg)
    if metadata["filters"] or metadata.get("dimension_separator", ".") != ".":
        msg = "The source has an unsupported chunk encoding."
        raise ValueError(msg)
    manifest_path = target / "acquisition.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["source"], manifest["variable"], manifest["year"]) != (
            SOURCE,
            variable,
            year,
        ):
            msg = "The existing target belongs to a different acquisition."
            raise ValueError(msg)
        return metadata, manifest["source_start_index"]
    if target.exists() and any(target.iterdir()):
        msg = "The target is nonempty but has no acquisition manifest."
        raise ValueError(msg)
    with xr.open_zarr(
        GCS_SOURCE, chunks=None, storage_options={"token": "anon"}
    ) as source:
        selection = source.indexes["time"].slice_indexer(
            f"{year}-01-01", f"{year}-12-31"
        )
        start, stop, step = selection.indices(source.sizes["time"])
        if step != 1 or stop == start:
            msg = "The requested year is not available."
            raise ValueError(msg)
        coordinates = xr.Dataset(
            coords={
                "time": source.time.isel(time=selection).load(),
                "latitude": source.latitude.load(),
                "longitude": source.longitude.load(),
            }
        )
        target.mkdir(parents=True, exist_ok=True)
        coordinates.to_zarr(target, mode="w", zarr_format=2, consolidated=False)
    metadata_local = dict(metadata, shape=[stop - start, *metadata["shape"][1:]])
    field = target / variable
    field.mkdir()
    (field / ".zarray").write_text(json.dumps(metadata_local, indent=2))
    (field / ".zattrs").write_text(json.dumps(attributes, indent=2))
    zarr.consolidate_metadata(str(target))
    manifest = {
        "source": SOURCE,
        "variable": variable,
        "year": year,
        "source_start_index": start,
        "shape": metadata_local["shape"],
        "dtype": metadata["dtype"],
        "chunks": metadata["chunks"],
        "decoded_bytes": int(np.prod(metadata_local["shape"]))
        * np.dtype(metadata["dtype"]).itemsize,
        "complete": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return metadata, start


def main(argv: list[str] | None = None) -> int:
    """Acquire an approved year, optionally stopping after a resumable pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--variable", default="2m_temperature")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--from-store", type=Path)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args(argv)
    if (
        args.workers < 1
        or args.step < 1
        or (args.stop_after is not None and args.stop_after < 1)
    ):
        parser.error("workers, step and stop-after must be positive")
    if args.from_store is not None:
        create_subset(args.from_store, args.target, args.variable, args.step)
        return 0
    metadata, start = initialize(args.target, args.variable, args.year)
    manifest_path = args.target / "acquisition.json"
    manifest = json.loads(manifest_path.read_text())
    count = manifest["shape"][0]
    limit = min(count, args.stop_after or count)
    if shutil.disk_usage(args.target).free < manifest["decoded_bytes"] + 1024**3:
        msg = "Insufficient free space for the decoded field plus a 1 GiB margin."
        raise OSError(msg)
    print(
        f"ERA5 {args.variable}, {args.year}: {count} hourly chunks, "
        f"{manifest['decoded_bytes'] / 1024**3:.2f} GiB decoded",
        flush=True,
    )
    began = time.perf_counter()
    downloaded_bytes = 0
    reused = 0
    with (
        (args.target / "chunks.jsonl").open("a") as log,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = [
            pool.submit(
                copy_chunk, args.target, args.variable, start + index, index, metadata
            )
            for index in range(limit)
        ]
        for completed, future in enumerate(as_completed(futures), 1):
            record = future.result()
            downloaded_bytes += 0 if record["reused"] else record["encoded_bytes"]
            reused += int(record["reused"])
            log.write(json.dumps(record) + "\n")
            log.flush()
            if completed % 32 == 0 or completed == limit:
                seconds = time.perf_counter() - began
                print(
                    f"{completed}/{limit} chunks verified; "
                    f"{downloaded_bytes / 1024**2:.1f} MiB downloaded; "
                    f"{seconds:.1f} s; reused={reused}",
                    flush=True,
                )
    manifest.update(
        complete=manifest["complete"] or limit == count,
        last_run_seconds=time.perf_counter() - began,
        last_run_downloaded_bytes=downloaded_bytes,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["complete"]:
        (args.target / "_SUCCESS").write_text(
            "All hourly chunks decoded and verified.\n"
        )
    print(
        "Complete."
        if manifest["complete"]
        else "Pilot complete; rerun without --stop-after to resume.",
        flush=True,
    )
    return 0


def create_subset(source: Path, target: Path, variable: str, step: int) -> None:
    """Select genuine timestamps by linking immutable hourly chunks into a new store."""
    if step < 1:
        msg = "The subset step must be positive."
        raise ValueError(msg)
    manifest = json.loads((source / "acquisition.json").read_text())
    if not manifest["complete"]:
        msg = "The source acquisition is incomplete."
        raise ValueError(msg)
    if manifest["variable"] != variable:
        msg = "The requested variable differs from the source acquisition."
        raise ValueError(msg)
    expected_source = str(source.resolve())
    target_manifest = target / "acquisition.json"
    if target_manifest.exists():
        existing = json.loads(target_manifest.read_text())
        if (existing.get("parent_store"), existing.get("step")) != (
            expected_source,
            step,
        ):
            msg = "The target belongs to a different subset."
            raise ValueError(msg)
        if existing["complete"]:
            return
    elif target.exists() and any(target.iterdir()):
        msg = "The target is nonempty but has no subset manifest."
        raise ValueError(msg)
    indices = range(0, manifest["shape"][0], step)
    if not target_manifest.exists():
        with xr.open_zarr(source, chunks=None) as dataset:
            coordinates = xr.Dataset(
                coords={
                    "time": dataset.time.isel(time=slice(None, None, step)).load(),
                    "latitude": dataset.latitude.load(),
                    "longitude": dataset.longitude.load(),
                }
            )
            target.mkdir(parents=True, exist_ok=True)
            coordinates.to_zarr(target, mode="w", zarr_format=2, consolidated=False)
        metadata = json.loads((source / variable / ".zarray").read_text())
        metadata["shape"][0] = len(indices)
        field = target / variable
        field.mkdir()
        (field / ".zarray").write_text(json.dumps(metadata, indent=2))
        shutil.copyfile(source / variable / ".zattrs", field / ".zattrs")
        zarr.consolidate_metadata(str(target))
        subset_manifest = dict(
            manifest,
            parent_store=expected_source,
            step=step,
            complete=False,
            shape=metadata["shape"],
            decoded_bytes=int(np.prod(metadata["shape"]))
            * np.dtype(metadata["dtype"]).itemsize,
        )
        target_manifest.write_text(json.dumps(subset_manifest, indent=2) + "\n")
    for local, original in enumerate(indices):
        destination = target / variable / f"{local}.0.0"
        if not destination.exists():
            original_path = source / variable / f"{original}.0.0"
            try:
                os.link(original_path, destination)
            except OSError:
                shutil.copyfile(original_path, destination)
    subset_manifest = json.loads(target_manifest.read_text())
    subset_manifest["complete"] = True
    target_manifest.write_text(json.dumps(subset_manifest, indent=2) + "\n")
    (target / "_SUCCESS").write_text("Subset chunks linked from verified source.\n")
    print(f"Prepared {len(indices)} snapshots at {target}.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
