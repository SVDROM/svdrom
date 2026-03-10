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

if __name__ == "__main__":
    dask_spill_folder = "/bask/projects/v/vjgo8416-dmd-ddwm/dask-spill"
    era5_slice_path = "data/era5_slice_2010-01-01_2022-12-31.zarr"
    svd_path = Path("data/svd.pkl")
    svd_hankel_path = Path("data/svd_hankel.pkl")
    scaler_path = Path("data/scaler.pkl")
    n_components = 40
    n_power_iter = 2
    n_oversamples = 15

    client = Client(processes=False, local_directory=dask_spill_folder)

    X = xr.open_dataarray(era5_slice_path, chunks="auto")
    X = X.sel(time=slice("2010", "2019"))

    print("Scaling data...")
    scaler = StandardScaler()
    X = scaler(X)
    scaler_path.parent.mkdir(exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler object saved to disk.")

    config.set(stack_coord_name="space")
    X = variable_spatial_stack(X, dims=("latitude", "longitude"))
    X = X.transpose(("space", "time"))

    print("Computing SVD of standard matrix...")
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
    print("SVD object saved to disk.")

    X_hankel = hankel_preprocessing(X)

    print("Computing SVD of Hankel pre-processed matrix...")
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
    print("Hankel SVD object saved to disk.")

    client.close()
