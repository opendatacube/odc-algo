# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2026 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from odc.algo._percentile import np_percentile, xr_quantile, xr_quantile_bands


def test_np_percentile():
    arr = np.array(
        [[0, 1, 4, 6, 8, 10, 15, 22, 25, 27], [3, 5, 6, 8, 9, 11, 15, 28, 31, 50]]
    )

    np.random.shuffle(arr[0, :])
    np.random.shuffle(arr[1, :])
    arr = arr.transpose()

    assert (np_percentile(arr, 0.5, 255) == np.array([8, 9])).all()
    assert (np_percentile(arr, 0.7, 255) == np.array([15, 15])).all()
    assert (np_percentile(arr, 1.0, 255) == np.array([27, 50])).all()
    assert (np_percentile(arr, 0.0, 255) == np.array([0, 3])).all()


@pytest.mark.parametrize("nodata", [255, 200, np.nan, -1])
def test_np_percentile_some_bad_data(nodata):
    arr = np.array(
        [
            [0, 1, 4, 6, 8, nodata, nodata, nodata, nodata, nodata],
            [3, 5, 6, 8, 9, 11, 15, 28, 31, 50],
        ]
    )

    np.random.shuffle(arr[0, :])
    np.random.shuffle(arr[1, :])
    arr = arr.transpose()

    assert (np_percentile(arr, 0.5, nodata) == np.array([4, 9])).all()
    assert (np_percentile(arr, 0.7, nodata) == np.array([6, 15])).all()
    assert (np_percentile(arr, 1.0, nodata) == np.array([8, 50])).all()
    assert (np_percentile(arr, 0.0, nodata) == np.array([0, 3])).all()


@pytest.mark.parametrize("nodata", [255, 200, np.nan])
def test_np_percentile_bad_data(nodata):
    arr = np.array(
        [
            [0, 1, nodata, nodata, nodata, nodata, nodata, nodata, nodata, nodata],
            [3, 5, 6, 8, 9, 11, 15, 28, 31, 50],
        ]
    )

    np.random.shuffle(arr[0, :])
    np.random.shuffle(arr[1, :])
    arr = arr.transpose()

    np.testing.assert_equal(np_percentile(arr, 0.5, nodata), np.array([nodata, 9]))
    np.testing.assert_equal(np_percentile(arr, 0.7, nodata), np.array([nodata, 15]))
    np.testing.assert_equal(np_percentile(arr, 1.0, nodata), np.array([nodata, 50]))
    np.testing.assert_equal(np_percentile(arr, 0.0, nodata), np.array([nodata, 3]))


@pytest.mark.parametrize("nodata", [255, 200, np.nan])
def test_np_percentile_min_valid_lowered(nodata):
    """min_valid can be lowered below the default of 3 to keep pixels with
    fewer valid observations, e.g. for short time series (HLS composites
    over a few scenes) where a hard 3-observation floor discards most
    pixels."""
    arr = np.array(
        [
            [0, 1, nodata, nodata, nodata, nodata, nodata, nodata, nodata, nodata],
            [3, 5, 6, 8, 9, 11, 15, 28, 31, 50],
        ]
    )

    np.random.shuffle(arr[0, :])
    np.random.shuffle(arr[1, :])
    arr = arr.transpose()

    # Default behaviour is unchanged: 2 valid observations is still below
    # the default floor of 3, so the first pixel stays nodata.
    np.testing.assert_equal(np_percentile(arr, 0.5, nodata), np.array([nodata, 9]))

    # With min_valid=2, the first pixel's 2 valid observations are enough.
    np.testing.assert_equal(
        np_percentile(arr, 0.0, nodata, min_valid=2), np.array([0, 3])
    )
    np.testing.assert_equal(
        np_percentile(arr, 0.5, nodata, min_valid=2), np.array([0, 9])
    )
    np.testing.assert_equal(
        np_percentile(arr, 0.7, nodata, min_valid=2), np.array([1, 15])
    )
    np.testing.assert_equal(
        np_percentile(arr, 1.0, nodata, min_valid=2), np.array([1, 50])
    )


@pytest.mark.parametrize("nodata", [255, 200, np.nan])
def test_np_percentile_min_valid_one(nodata):
    """min_valid=1 keeps a pixel with a single valid observation."""
    arr = np.array(
        [
            [7, nodata, nodata, nodata, nodata, nodata, nodata, nodata, nodata, nodata],
            [3, 5, 6, 8, 9, 11, 15, 28, 31, 50],
        ]
    )
    arr = arr.transpose()

    np.testing.assert_equal(
        np_percentile(arr, 0.5, nodata, min_valid=1), np.array([7, 9])
    )
    # A single observation is still below min_valid=2, so it stays nodata.
    np.testing.assert_equal(
        np_percentile(arr, 0.5, nodata, min_valid=2), np.array([nodata, 9])
    )


@pytest.mark.parametrize("nodata", [255, 200, np.nan, -1])
@pytest.mark.parametrize("use_dask", [False, True])
def test_xr_quantile_bands(nodata, use_dask):
    band_1 = np.random.randint(0, 100, size=(10, 100, 200)).astype(type(nodata))
    band_2 = np.random.randint(0, 100, size=(10, 100, 200)).astype(type(nodata))

    band_1[np.random.random(size=band_1.shape) > 0.5] = nodata
    band_2[np.random.random(size=band_1.shape) > 0.5] = nodata

    true_results = {}
    true_results["band_1_pc_20"] = np_percentile(band_1, 0.2, nodata)
    true_results["band_2_pc_20"] = np_percentile(band_2, 0.2, nodata)
    true_results["band_1_pc_60"] = np_percentile(band_1, 0.6, nodata)
    true_results["band_2_pc_60"] = np_percentile(band_2, 0.6, nodata)

    if use_dask:
        band_1 = da.from_array(band_1, chunks=(2, 20, 20))
        band_2 = da.from_array(band_2, chunks=(2, 20, 20))

    attrs = {"test": "attrs"}
    coords = {
        "x": np.linspace(10, 20, band_1.shape[2]),
        "y": np.linspace(0, 5, band_1.shape[1]),
        "t": np.linspace(0, 5, band_1.shape[0]),
    }

    data_vars = {
        "band_1": (("t", "y", "x"), band_1),
        "band_2": (("t", "y", "x"), band_2),
    }

    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    output = xr_quantile_bands(dataset, [0.2, 0.6], nodata).compute()

    for band in output.keys():
        np.testing.assert_equal(output[band], true_results[band])


@pytest.mark.parametrize("nodata", [255, 200, np.nan, -1])
@pytest.mark.parametrize("use_dask", [False, True])
def test_xr_quantile(nodata, use_dask):
    band_1 = np.random.randint(0, 100, size=(10, 100, 200)).astype(type(nodata))
    band_2 = np.random.randint(0, 100, size=(10, 100, 200)).astype(type(nodata))

    band_1[np.random.random(size=band_1.shape) > 0.5] = nodata
    band_2[np.random.random(size=band_1.shape) > 0.5] = nodata

    true_results = {}
    true_results["band_1"] = np.stack(
        [np_percentile(band_1, 0.2, nodata), np_percentile(band_1, 0.6, nodata)], axis=0
    )
    true_results["band_2"] = np.stack(
        [np_percentile(band_2, 0.2, nodata), np_percentile(band_2, 0.6, nodata)], axis=0
    )

    if use_dask:
        band_1 = da.from_array(band_1, chunks=(2, 20, 20))
        band_2 = da.from_array(band_2, chunks=(2, 20, 20))

    attrs = {"test": "attrs"}
    coords = {
        "x": np.linspace(10, 20, band_1.shape[2]),
        "y": np.linspace(0, 5, band_1.shape[1]),
        "t": np.linspace(0, 5, band_1.shape[0]),
    }

    data_vars = {
        "band_1": xr.DataArray(band_1, dims=("t", "y", "x"), attrs={"test_attr": 1}),
        "band_2": xr.DataArray(band_2, dims=("t", "y", "x"), attrs={"test_attr": 2}),
    }

    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    output = xr_quantile(dataset, [0.2, 0.6], nodata).compute()

    for band in output.keys():
        np.testing.assert_equal(output[band], true_results[band])


@pytest.mark.parametrize("use_dask", [False, True])
def test_xr_quantile_bands_min_valid(use_dask):
    """min_valid is threaded through to np_percentile for every band/quantile."""
    nodata = 255
    band_1 = np.random.randint(0, 100, size=(10, 100, 200)).astype(np.uint8)
    band_1[np.random.random(size=band_1.shape) > 0.5] = nodata

    true_results = {
        "band_1_pc_20": np_percentile(band_1, 0.2, nodata, min_valid=1),
        "band_1_pc_60": np_percentile(band_1, 0.6, nodata, min_valid=1),
    }

    if use_dask:
        band_1 = da.from_array(band_1, chunks=(2, 20, 20))

    dataset = xr.Dataset(data_vars={"band_1": (("t", "y", "x"), band_1)})
    output = xr_quantile_bands(dataset, [0.2, 0.6], nodata, min_valid=1).compute()

    for band in output.keys():
        np.testing.assert_equal(output[band], true_results[band])


@pytest.mark.parametrize("use_dask", [False, True])
def test_xr_quantile_min_valid(use_dask):
    """min_valid is threaded through to np_percentile for every band/quantile."""
    nodata = 255
    band_1 = np.random.randint(0, 100, size=(10, 100, 200)).astype(np.uint8)
    band_1[np.random.random(size=band_1.shape) > 0.5] = nodata

    true_results = {
        "band_1": np.stack(
            [
                np_percentile(band_1, 0.2, nodata, min_valid=1),
                np_percentile(band_1, 0.6, nodata, min_valid=1),
            ],
            axis=0,
        )
    }

    if use_dask:
        band_1 = da.from_array(band_1, chunks=(2, 20, 20))

    dataset = xr.Dataset(data_vars={"band_1": (("t", "y", "x"), band_1)})
    output = xr_quantile(dataset, [0.2, 0.6], nodata, min_valid=1).compute()

    for band in output.keys():
        np.testing.assert_equal(output[band], true_results[band])
