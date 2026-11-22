from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pooch
import pytest
import xarray as xr


@pytest.fixture()
def intro_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run the notebook's numerical workflow on a small offline weather fixture."""
    rng = np.random.default_rng(42)
    values = (275 + rng.normal(size=(64, 6)) @ rng.normal(size=(6, 12))).T.reshape(
        12, 8, 8
    )
    sample = xr.Dataset(
        {"air": (("time", "lat", "lon"), values)},
        coords={
            "time": pd.date_range("2004-01-01", periods=12, freq="6h"),
            "lat": np.linspace(75, 15, 8, dtype="float32"),
            "lon": np.arange(0, 360, 45, dtype="float32"),
        },
    )

    def retrieve_annual(url: str, path: Path, fname: str, **_kwargs: Any) -> str:
        """Replace each NOAA download with a local annual NetCDF fixture."""
        assert url.startswith(
            "https://downloads.psl.noaa.gov/"
        ), "Expected the documented NOAA source."
        year = int(fname.split(".")[2])
        annual = sample.assign_coords(
            time=pd.date_range(f"{year}-01-01", periods=12, freq="6h")
        )
        path.mkdir(parents=True, exist_ok=True)
        destination = path / fname
        annual.to_netcdf(destination, engine="h5netcdf")
        return str(destination)

    monkeypatch.setattr(pooch, "retrieve", retrieve_annual)
    monkeypatch.chdir(tmp_path)
    notebook_path = Path(__file__).resolve().parents[1] / "demos" / "intro.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace: dict[str, Any] = {"__name__": "__main__"}
    # Restore the caller's scheduler settings after the notebook configures Dask.
    with dask.config.set(
        scheduler="synchronous", num_workers=1, **{"array.chunk-size": "128 MiB"}
    ):
        for cell in notebook["cells"]:
            if "intro-core" in cell.get("metadata", {}).get("tags", []):
                source = "".join(cell["source"])
                exec(compile(source, str(notebook_path), "exec"), namespace)
    namespace["expected_snapshot"] = (values[0].astype("float32") - np.float32(273.15))[
        :, [4, 5, 6, 7, 0, 1, 2, 3]
    ]
    return namespace


def test_intro_uses_public_pod_reconstruction(intro_run: dict[str, Any]) -> None:
    """Check lazy input, physical-unit normalization, and restored coordinates."""
    result = intro_run
    assert isinstance(result["X"].data, da.Array), "Expected Dask-backed POD input."
    assert isinstance(
        result["reconstruction"].data, da.Array
    ), "Expected public reconstruction to stay lazy for the larger intro."
    assert (
        result["X"].sizes["time"] == len(result["YEARS"]) * 12
    ), "Expected the model to include every snapshot from every annual source."
    np.testing.assert_array_equal(
        result["temperature"].lon.to_numpy(),
        np.arange(-180, 180, 45),
        err_msg="Expected global longitudes to wrap and sort into increasing order.",
    )
    np.testing.assert_allclose(
        result["temperature"].isel(time=0).to_numpy(),
        result["expected_snapshot"],
        err_msg="Longitude wrapping should preserve the corresponding temperatures.",
    )
    expected = (
        result["mean"][:, None] + result["modes"] @ result["coefficients"]
    ).T.reshape(result["reconstruction"].shape)
    np.testing.assert_allclose(
        result["reconstruction"].to_numpy(),
        expected,
        atol=1e-4,
        err_msg="Stored factors should reproduce the public POD reconstruction.",
    )
    for coordinate in ("time", "lat", "lon"):
        np.testing.assert_array_equal(
            result["reconstruction"][coordinate].to_numpy(),
            result["temperature"][coordinate].to_numpy(),
            err_msg=f"Expected public reconstruction to retain {coordinate} labels.",
        )


def test_intro_saved_encoding_is_self_contained(intro_run: dict[str, Any]) -> None:
    """Check that reopened factors include the mean, labels, and declared rank."""
    result = intro_run
    assert isinstance(
        result["decoded"].data, da.Array
    ), "Expected decoding to produce time chunks rather than a full NumPy array."
    assert (
        max(result["decoded"].chunksizes["time"]) <= result["TIME_CHUNK"]
    ), "Expected decoded time chunks to stay within the configured block length."
    with np.load(result["encoding_path"], allow_pickle=False) as saved:
        assert set(saved.files) == {
            "mean",
            "modes",
            "coefficients",
            "time",
            "lat",
            "lon",
        }, "Expected all factors and coordinates needed for independent decoding."
        assert (
            saved["modes"].shape[1] == result["RANK"]
        ), "Expected only the selected rank in the archive."
        for name in ("mean", "modes", "coefficients"):
            assert saved[name].dtype == np.dtype(
                "float32"
            ), f"Expected float32 {name} as documented."
        for coordinate in ("time", "lat", "lon"):
            np.testing.assert_array_equal(
                result["decoded"][coordinate].to_numpy(),
                saved[coordinate],
                err_msg=f"Expected saved {coordinate} labels to survive decoding.",
            )
    assert (
        result["archive_bytes"] > result["factor_bytes"] + result["coordinate_bytes"]
    ), "Expected actual file size to include NPZ overhead beyond array payloads."


def test_intro_metrics_score_actual_decoding(intro_run: dict[str, Any]) -> None:
    """Verify byte accounting, full-field errors, and the fluctuation baseline."""
    result = intro_run
    metrics = result["metrics"]
    assert isinstance(
        result["reference_matrix"], da.Array
    ), "Expected full-input scoring to use lazy reductions."
    rmse = metrics["RMSE (°C)"].to_numpy()
    assert np.all(
        np.diff(rmse) <= 1e-5
    ), "Expected error to decrease as rank increases."
    selected = metrics.loc[result["RANK"]]
    np.testing.assert_allclose(
        selected["RMSE (°C)"],
        result["decoded_rmse"],
        atol=1e-7,
        err_msg="Rank curve should score the same decoding as the reopened archive.",
    )
    np.testing.assert_allclose(
        selected["Factor payload (KiB)"] * 2**10,
        result["factor_bytes"],
        err_msg="Factor-size estimates should include modes, coefficients, and mean.",
    )
    np.testing.assert_allclose(
        metrics.iloc[-1]["Retained fluctuation energy (%)"] / 100,
        result["pod"].explained_energy_ratio.sum(),
        atol=2e-5,
        err_msg="Direct errors should agree with POD's centered energy fractions.",
    )
