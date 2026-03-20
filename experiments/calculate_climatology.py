import logging
from pathlib import Path

import xarray as xr
from dask.distributed import Client

from svdrom.weather_utils import compute_climatology

logging.basicConfig(format="%(message)s", level=logging.INFO)

if __name__ == "__main__":
    start_year = 2010
    end_year = 2019
    dask_spill_folder = "/bask/projects/v/vjgo8416-dmd-ddwm/dask-spill"
    base_path = Path("data")
    era5_slice_path = base_path / "era5_slice_2010-01-01_2022-12-31.zarr"
    climatology_path = base_path / f"climatology_{start_year}-{end_year}.zarr"

    client = Client(processes=False, local_directory=dask_spill_folder)
    logging.info("Dask dashboard: %s", client.dashboard_link)

    X = xr.open_dataarray(era5_slice_path, chunks="auto")
    X = X.sel(time=slice(str(start_year), str(end_year)))

    logging.info("Calculating climatology...")
    climatology = compute_climatology(X)
    logging.info("Done.")

    logging.info("Rechunking...")
    climatology = climatology.chunk(
        {
            "dayofyear": 3,
            "hour": -1,
            "latitude": -1,
            "longitude": -1,
        }
    )
    logging.info("Done.")
    logging.info("Dask blocks: %s = %s", climatology.dims, climatology.data.numblocks)

    logging.info("Saving to disk...")
    climatology_path.parent.mkdir(exist_ok=True)
    climatology.to_zarr(
        climatology_path,
        zarr_format=2,
        mode="w",
    )
    logging.info("Done.")

    client.close()
