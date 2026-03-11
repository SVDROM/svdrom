import logging
from pathlib import Path

import xarray as xr
from dask.distributed import Client

from ..demos.utils import compute_climatology

logging.basicConfig(format="%(message)s", level=logging.INFO)

if __name__ == "__main__":
    dask_spill_folder = "/bask/projects/v/vjgo8416-dmd-ddwm/dask-spill"
    era5_slice_path = "data/era5_slice_2010-01-01_2022-12-31.zarr"
    climatology_path = Path("data/climatology_JanFeb2020.zarr")

    client = Client(processes=False, local_directory=dask_spill_folder)
    logging.info("Dask dashboard: %s", client.dashboard_link)

    X = xr.open_dataarray(era5_slice_path, chunks="auto")
    X = X.sel(time=slice("2010", "2019"))

    logging.info("Computing climatology...")
    climatology = compute_climatology(X, year=2020, months=[1, 2])
    logging.info("Done.")

    logging.info("Saving to disk...")
    climatology_path.parent.mkdir(exist_ok=True)
    climatology.to_zarr(
        climatology_path,
        zarr_format=2,
        mode="w",
    )
    logging.info("Done.")

    client.close()
