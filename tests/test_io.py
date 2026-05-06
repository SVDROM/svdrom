import os
import shutil

import netCDF4  # noqa: F401
import pytest
import xarray as xr
from make_test_data import DataGenerator

from svdrom.io import open_dataarray, open_dataset


class TestDataHandler:
    """
    Test suite for verifying the functionality of the DataHandler class with
    different file formats.

    This class uses pytest to test reading of xarray.Dataset and xarray.DataArray
    objects using both Zarr and NetCDF formats. It sets up a temporary directory
    for test data, generates synthetic datasets and data arrays, and ensures that
    the DataHandler can correctly open and reproduce the original data.

    Class Attributes:
        data_dir (str): Path to the temporary directory used for storing test files.
        data_generator (DataGenerator): Utility for generating synthetic xarray
            datasets and data arrays.
        data_handler (DataHandler): The handler being tested for reading datasets and
            data arrays.

    Methods:
        setup_class: Class-level setup to initialize directories and test utilities.
        teardown_class: Class-level teardown to clean up test directories.
        _make_test_data: Pytest fixture to generate synthetic test data before each
            test.
        test_open_formats: Parametrized test to check reading of datasets/data arrays
            in both Zarr and NetCDF formats using the DataHandler class.
    """

    @classmethod
    def setup_class(cls):
        cls.data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data", "io"
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

        ds = open_dataset(ds_path)
        da = open_dataarray(da_path)

        assert isinstance(
            ds, xr.Dataset
        ), f"Expected xarray.Dataset, got {type(ds).__name__}."
        assert isinstance(
            da, xr.DataArray
        ), f"Expected xarray.DataArray, got {type(da).__name__}."

        assert self.data_generator.ds.equals(
            ds
        ), f"Xarray Datasets should be equal for {filetype}."
        assert self.data_generator.da.equals(
            da
        ), f"Xarray DataArrays should be equal for {filetype}."
