import numpy as np
import xarray as xr


def compute_rmse(
    ground_truth: xr.DataArray,
    prediction: xr.DataArray,
    lat_weighting: bool = True,
    dims: str | tuple[str, ...] = ("latitude", "longitude"),
) -> xr.DataArray:
    """Compute the Root Mean Squared Error (RMSE) of a prediction,
    averaging along the specified dimension(s).

    Parameters
    ----------
    ground_truth: xr.DataArray
        A dask- or numpy-backed DataArray containing the ground truth
        data.
    prediction: xr.DataArray
        A dask- or numpy-backed DataArray containing the prediction (
        a reconstruction or a forecast). The prediction and ground truth
        must be defined on the same spatio-temporal grid.
    lat_weights: bool, optional
        Whether to apply a weighting function so that spatial locations
        in a lat/lon grid closer to the Equator receive a larger weight
        than those closer to the poles. Default is True.
    dims: str | tuple[str], optional
        Dimensions along which to average the RMSE. The default is
        ("latitude", "longitude"), which would return the RMSE as a function
        of prediction time (assuming a single variable and pressure level).

    Returns
    -------
    xr.DataArray:
        A numpy-backed DataArray containing the calculated RMSE score.

    Notes
    -----
    If the inputs are dask-backed DataArrays, the function will build the
    task graph and trigger the computation at the end. You might want to
    set up a multi-threading Dask cluster before calling the function.
    """
    if set(ground_truth.dims) != set(prediction.dims):
        msg = "The dimensions of the ground truth data and " "prediction don't match."
        raise ValueError(msg)
    prediction = prediction.real  # keep only real part of the prediction
    rmse = ground_truth.copy(data=(ground_truth - prediction) ** 2)
    if lat_weighting:
        lat_weights_np = np.cos(np.deg2rad(ground_truth.latitude.values))
        lat_weights_np = lat_weights_np / np.mean(lat_weights_np)
        lat_weights_dict = {}
        lat_weights_dict["data"] = lat_weights_np
        lat_weights_dict["dims"] = "latitude"
        lat_weights_dict["latitude"] = {"data": rmse.latitude, "dims": ("latitude")}
        lat_weights = xr.DataArray.from_dict(lat_weights_dict)
        rmse *= lat_weights
    rmse = rmse.mean(dim=dims)
    rmse = rmse.clip(min=0)
    rmse = xr.ufuncs.sqrt(rmse)
    return rmse.compute()
