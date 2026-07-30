"""Convert the airfoilLES midspan HDF5 snapshots into a single chunked Zarr store.

Each snapshot file (airfoilLES_tNNNNN.h5) holds flat (npoint,) float32 arrays
ux/uy/uz for one timestep, over an unstructured point cloud shared across all
snapshots (defined in airfoilLES_grid.h5). This script stacks all snapshots
into (time, point) Zarr arrays, chunked only along time, so the result opens
cleanly as a Dask-backed xarray.Dataset via `xr.open_zarr(..., chunks={})`.
"""

import glob
import re
import time

import dask.array as da
import h5py
import numpy as np
import xarray as xr

GRID_PATH = "data/airfoilLES/airfoilLES_grid.h5"
SNAPSHOT_DIR = "data/airfoilLES/airfoilLES_midspan"
STORE_PATH = "data/airfoilLES/airfoilLES_midspan.zarr"
CHUNK_TIME = 100  # timesteps per chunk; chunking is along time only

VAR_NAMES = ["ux", "uy", "uz"]


def snapshot_index(path):
    return int(re.search(r"t(\d+)\.h5$", path).group(1))


def main():
    files = sorted(glob.glob(f"{SNAPSHOT_DIR}/*.h5"), key=snapshot_index)
    times = np.array([snapshot_index(f) for f in files])
    ntime = len(files)

    with h5py.File(GRID_PATH, "r") as f:
        x = f["x"][:]
        y = f["y"][:]
        w = f["w"][:]
    npoint = x.shape[0]

    print(f"{ntime} snapshots, {npoint} points, chunk_time={CHUNK_TIME}")

    # 1. Write metadata only: correct shape/dtype/chunks for each variable,
    #    no data computed yet (compute=False), plus the real coordinates.
    skeleton = xr.Dataset(
        {
            var: (
                ("time", "point"),
                da.empty((ntime, npoint), chunks=(CHUNK_TIME, npoint), dtype="float32"),
            )
            for var in VAR_NAMES
        },
        coords={
            "time": times,
            "point": np.arange(npoint),
            "x": ("point", x),
            "y": ("point", y),
            "w": ("point", w),
        },
    )
    skeleton.to_zarr(STORE_PATH, mode="w", compute=False)

    # 2. Fill in real data one time-chunk at a time via region writes, so
    #    memory use stays bounded to one chunk regardless of dataset size.
    n_chunks = (ntime + CHUNK_TIME - 1) // CHUNK_TIME
    t0 = time.time()
    for i, start in enumerate(range(0, ntime, CHUNK_TIME)):
        stop = min(start + CHUNK_TIME, ntime)
        n = stop - start

        buffers = {var: np.empty((n, npoint), dtype="float32") for var in VAR_NAMES}
        for j, path in enumerate(files[start:stop]):
            with h5py.File(path, "r") as f:
                for var in VAR_NAMES:
                    buffers[var][j] = f[var][:]

        chunk_ds = xr.Dataset(
            {var: (("time", "point"), buffers[var]) for var in VAR_NAMES},
            coords={"time": times[start:stop]},
        )
        chunk_ds.to_zarr(STORE_PATH, region={"time": slice(start, stop)})

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (n_chunks - i - 1) / rate if rate > 0 else float("nan")
        print(
            f"chunk {i + 1}/{n_chunks} "
            f"(timesteps {times[start]}-{times[stop - 1]}) written "
            f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]"
        )

    print(f"done in {time.time() - t0:.0f}s -> {STORE_PATH}")


if __name__ == "__main__":
    main()
