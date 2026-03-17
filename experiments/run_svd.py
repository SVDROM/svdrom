import logging
import pickle
from pathlib import Path

import xarray as xr
from dask.distributed import Client

import svdrom.config as config
from svdrom.preprocessing import (
    StandardScaler,
    hankel_preprocessing,
    variable_spatial_stack,
)
from svdrom.svd import TruncatedSVD

logging.basicConfig(format="%(message)s", level=logging.INFO)

if __name__ == "__main__":
    start_date = "2010-07-01"
    end_date = "2020-06-30"
    subfolder = start_date + "_" + end_date
    dask_spill_folder = "/bask/projects/v/vjgo8416-dmd-ddwm/dask-spill"
    base_path = Path("data")
    era5_slice_path = base_path / "era5_slice_2010-01-01_2022-12-31.zarr"
    svd_path = base_path / subfolder / "svd.pkl"
    svd_hankel_path = base_path / subfolder / "svd_hankel.pkl"
    scaler_path = base_path / subfolder / "scaler.pkl"
    n_components = 40
    n_power_iter = 2
    n_oversamples = 15

    client = Client(processes=False, local_directory=dask_spill_folder)
    logging.info("Dask dashboard: %s", client.dashboard_link)

    X = xr.open_dataarray(era5_slice_path, chunks="auto")
    X = X.sel(time=slice(start_date, end_date))

    logging.info("Scaling data...")
    scaler = StandardScaler()
    X = scaler(X)
    scaler_path.parent.mkdir(exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logging.info("Scaler object saved to disk.")

    config.set(stack_coord_name="space")
    X = variable_spatial_stack(X, dims=("latitude", "longitude"))
    X = X.transpose("space", "time")

    logging.info("Computing SVD of standard matrix...")
    svd = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        rechunk=True,
        compute_var_ratio=True,
    )
    svd.fit(X, n_power_iter=n_power_iter, n_oversamples=n_oversamples)
    svd_path.parent.mkdir(exist_ok=True)
    with open(svd_path, "wb") as f:
        pickle.dump(svd, f)
    logging.info("SVD object saved to disk.")

    X_hankel = hankel_preprocessing(X)

    logging.info("Computing SVD of Hankel pre-processed matrix...")
    svd_hankel = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        rechunk=True,
        compute_var_ratio=True,
    )
    svd_hankel.fit(X_hankel, n_power_iter=n_power_iter, n_oversamples=n_oversamples)
    svd_hankel_path.parent.mkdir(exist_ok=True)
    with open(svd_hankel_path, "wb") as f:
        pickle.dump(svd_hankel, f)
    logging.info("Hankel SVD object saved to disk.")

    client.close()
