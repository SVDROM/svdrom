from __future__ import annotations

import os
import shutil

import netCDF4  # noqa: F401
import pytest
import xarray as xr
from make_test_data import DataGenerator


class TestDataLoading:
    """
    Integration test suite verifying that the project environment can read the
    file formats users feed into SVD-ROM.

    Synthetic datasets and data arrays are written to a temporary directory in
    Zarr, NetCDF4 and HDF5 format, and read back with Xarray's native I/O
    functions (``xr.open_dataset`` / ``xr.open_dataarray``). This exercises the
    installed backends (zarr, netCDF4, h5netcdf) and the Dask integration.

    Class Attributes:
        data_dir (str): Path to the temporary directory used for storing test files.
        data_generator (DataGenerator): Utility for generating synthetic xarray
            datasets and data arrays.

    Methods:
        setup_class: Class-level setup to initialize directories and test utilities.
        teardown_class: Class-level teardown to clean up test directories.
        _make_test_data: Pytest fixture to generate synthetic test data before each
            test.
        test_open_formats: Parametrized test checking that datasets/data arrays
            round-trip through Zarr, NetCDF4 and HDF5 with Xarray.
    """

    @classmethod
    def setup_class(cls):
        cls.data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data", "data_loading"
        )
        cls.data_generator = DataGenerator()

        # create clean directories
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)
        os.makedirs(cls.data_dir)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.data_dir)

    @pytest.fixture(autouse=True)
    def _make_test_data(self):
        print("Generating synthetic Xarray.DataSet...")
        self.data_generator.generate_dataset()
        print("Generating synthetic Xarray.DataArray...")
        self.data_generator.generate_dataarray()
        return

    @pytest.mark.parametrize(
        ("filetype", "ds_ext", "da_ext", "ds_save", "da_save"),
        [
            (
                "zarr",
                "dataset.zarr",
                "dataarray.zarr",
                lambda ds, path: ds.to_zarr(path, zarr_format=2),
                lambda da, path: da.to_zarr(path, zarr_format=2),
            ),
            (
                "netcdf",
                "dataset.nc",
                "dataarray.nc",
                lambda ds, path: ds.to_netcdf(path, format="NETCDF4"),
                lambda da, path: da.to_netcdf(path, format="NETCDF4"),
            ),
            (
                "h5",
                "dataset.h5",
                "dataarray.h5",
                lambda ds, path: ds.to_netcdf(path),
                lambda da, path: da.to_netcdf(path),
            ),
        ],
    )
    def test_open_formats(self, filetype, ds_ext, da_ext, ds_save, da_save):
        ds_path = os.path.join(self.data_dir, ds_ext)
        da_path = os.path.join(self.data_dir, da_ext)
        ds_save(self.data_generator.ds, ds_path)
        da_save(self.data_generator.da, da_path)

        ds = xr.open_dataset(ds_path, chunks="auto")
        da = xr.open_dataarray(da_path, chunks="auto")

        assert isinstance(
            ds, xr.Dataset
        ), f"Expected xarray.Dataset, got {type(ds).__name__}."
        assert isinstance(
            da, xr.DataArray
        ), f"Expected xarray.DataArray, got {type(da).__name__}."

        assert ds.chunks, f"Expected a Dask-backed Dataset for {filetype}."
        assert da.chunks, f"Expected a Dask-backed DataArray for {filetype}."

        assert self.data_generator.ds.equals(
            ds
        ), f"Xarray Datasets should be equal for {filetype}."
        assert self.data_generator.da.equals(
            da
        ), f"Xarray DataArrays should be equal for {filetype}."
