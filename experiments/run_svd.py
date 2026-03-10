import xarray as xr
from dask.distributed import Client
from svdrom.preprocessing import StandardScaler
from svdrom.svd import TruncatedSVD

if __name__ = "__main__":
    
    client = Client(processes=False, local_directory="/bask/projects/v/vjgo8416-dmd-ddwm/dask-spill")
    X = xr.open_dataarray("demos/data/era5_slice_2010-01-01_2022-12-31.zarr", engine="zarr")
    scaler =
