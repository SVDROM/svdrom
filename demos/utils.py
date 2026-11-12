"""Some utility functions for the demos."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import xarray as xr
from dask.diagnostics import ProgressBar


def write_to_zarr(
    da: xr.DataArray, filename: str, folder: str = "data", append_dim: str = "time"
) -> None:
    """Writes an xarray DataArray to a Zarr file, creating the file if it does not,
    exist or appending along the `append_dim` dimension if it does.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to be saved to disk.
    filename : str
        The name of the Zarr file to write to (within the specified folder).
    folder : str, optional
        The directory where the Zarr file will be stored. Defaults to "data/".
    append_dim: str, optional
        Dimension along which to append data if the file already exists.
        Defaults to "time".

    Notes
    -----
    - If the specified Zarr file does not exist, it will be created.
    - If the file exists, data will be appended along the 'time' dimension.
    - The function uses Dask for parallelized writing and displays a progress bar.
    """

    os.makedirs(folder, exist_ok=True)

    if not os.path.exists(os.path.join(folder, filename)):
        write_job = da.to_zarr(
            os.path.join(folder, filename),
            zarr_format=2,
            compute=False,  # return a Dask delayed object
        )
    else:
        write_job = da.to_zarr(
            os.path.join(folder, filename),
            zarr_format=2,
            append_dim=append_dim,
            compute=False,  # return a Dask delayed object
        )
    with ProgressBar():
        print("Saving to disk...")
        write_job.compute()
    print("Saved to disk.")


def display_size(arr: xr.Dataset | xr.DataArray) -> None:
    """Print Dataset or DataArray size in GiB."""
    array_size = arr.nbytes / (1024**3)
    print(f"Array size: {array_size:.2f} GiB")


def resample_array(
    arr: xr.Dataset | xr.DataArray, hours: int
) -> xr.Dataset | xr.DataArray:
    """Resample a xarray Dataset or DataArray along the time dimension,
    in hours.
    """
    return arr.resample(time=timedelta(hours=hours)).nearest()


def generate_time_ranges(start_date: str, end_date: str, num_periods: int) -> list:
    """Generates a list of time ranges between a start and end date, divided into
    a specified number of periods.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        num_periods (int): The number of periods to divide the date range into.

    Returns:
        list: A list of lists, where each inner list contains the start and end date
        (as strings in 'YYYY-MM-DD' format) for each period.

    Example:
        >>> generate_time_ranges("2024-01-01", "2024-01-10", 3)
        [['2024-01-01', '2024-01-04'],
         ['2024-01-05', '2024-01-08'],
         ['2024-01-09', '2024-01-10']]
    """
    datetime_format = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, datetime_format)
    end_date = datetime.strptime(end_date, datetime_format)
    time_delta = (end_date - start_date) / num_periods
    time_ranges = []
    start = start_date
    for _period in range(num_periods):
        end = start + time_delta
        time_ranges.append(
            [
                datetime.strftime(start, datetime_format),
                datetime.strftime(end, datetime_format),
            ]
        )
        start = end + timedelta(days=1)
    time_ranges[-1][-1] = datetime.strftime(end_date, datetime_format)
    return time_ranges
