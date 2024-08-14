import time
import numpy as np
import pytest
from odc.algo._geomedian_impl import geomedian


@pytest.fixture
def arr(kwargs):
    xx = np.random.randint(-100, 10_000, size=(100, 100, 10, 50), dtype=np.int16)
    mask = np.random.random((100, 100, 10, 50)) < 0.01
    xx[mask] = kwargs["nodata"]
    return xx


@pytest.fixture
def kwargs():
    return {
        "maxiters": 1000,
        "eps": 1e-5,
        "num_threads": 1,
        "nodata": -999,
        "scale": 1 / 10_000,
    }


@pytest.fixture
def kwargs_parallel():
    return {
        "maxiters": 1000,
        "eps": 1e-5,
        "num_threads": 4,
        "nodata": -999,
        "scale": 1 / 10_000,
    }


def test_benchmark_int16(benchmark, arr, kwargs):
    benchmark(geomedian, arr, **kwargs)


def test_benchmark_parallel_int16(benchmark, arr, kwargs_parallel):
    benchmark(geomedian, arr, **kwargs_parallel)


def test_accuracy(arr, kwargs):
    arr_f32 = arr.astype(np.float32)
    arr_f32[arr == kwargs["nodata"]] = np.nan
    gm_1, mads_1 = geomedian(arr, **kwargs)
    gm_2, mads_2 = geomedian(arr_f32, **kwargs)

    assert (np.abs(gm_1 - gm_2) <= 0.5).all()
    assert (mads_1 == mads_2).all()


def test_novalid_measurements(arr, kwargs):
    arr_bad = arr.copy()
    arr_bad[1, 2, :, :] = kwargs["nodata"]

    gm_1, mads_1 = geomedian(arr, **kwargs)
    gm_2, mads_2 = geomedian(arr_bad, **kwargs)

    assert (gm_1[:1, :2, :] == gm_2[:1, :2, :]).all()
    assert (gm_1[2:, 3:, :] == gm_2[2:, 3:, :]).all()
    assert (gm_1[1, :2, :] == gm_2[1, :2, :]).all()
    assert (gm_1[1, 3:, :] == gm_2[1, 3:, :]).all()
    assert (gm_1[1, :2, :] == gm_2[1, :2, :]).all()
    assert (gm_1[:1, 2, :] == gm_2[:1, 2, :]).all()
    assert (gm_1[2:, 2, :] == gm_2[2:, 2, :]).all()
    assert (gm_2[1, 2, :] == kwargs["nodata"]).all()

    assert (mads_1[:1, :2, :] == mads_2[:1, :2, :]).all()
    assert (mads_1[2:, 3:, :] == mads_2[2:, 3:, :]).all()
    assert (mads_1[1, :2, :] == mads_2[1, :2, :]).all()
    assert (mads_1[1, 3:, :] == mads_2[1, 3:, :]).all()
    assert (mads_1[1, :2, :] == mads_2[1, :2, :]).all()
    assert (mads_1[:1, 2, :] == mads_2[:1, 2, :]).all()
    assert (mads_1[2:, 2, :] == mads_2[2:, 2, :]).all()
    assert np.isnan(mads_2[1, 2, :]).all()


def test_offset(arr, kwargs):
    arr_f32 = arr.astype(np.float32)
    arr_f32[arr == kwargs["nodata"]] = np.nan

    args_1 = kwargs.copy()
    args_1["offset"] = np.float32(100.0)

    gm_1, mads_1 = geomedian(arr, **args_1)
    gm_2, mads_2 = geomedian(arr_f32, **args_1)

    assert (np.abs(gm_1 - gm_2) <= 0.5).all()
    assert (mads_1 == mads_2).all()
