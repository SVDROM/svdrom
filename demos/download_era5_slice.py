import xarray as xr
from utils import (
    generate_time_ranges,
    resample_array,
    write_to_zarr,
)

era5_path = "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
era5 = xr.open_zarr(era5_path, chunks={"time": "auto"})

variable = "temperature"
level = 850
hours = 6
start_date, end_date = "2010-01-01", "2022-12-31"

era5 = era5[variable].sel(level=level, time=slice(start_date, end_date))
era5 = resample_array(era5, hours=hours)

time_periods = generate_time_ranges(start_date, end_date, num_periods=4)
era5_list = []
for period in time_periods:
    era5_slice = era5.sel(time=slice(period[0], period[1]))
    era5_list.append(era5_slice)

filename = f"era5_slice_{start_date}_{end_date}.zarr"
folder = "/bask/projects/v/vjgo8416-dmd-ddwm/svdrom/demos/data"

for i, era5_slice in enumerate(era5_list):
    print(f"Downloading slice {i} out of {len(era5_list)}...")
    write_to_zarr(era5_slice, filename, folder)