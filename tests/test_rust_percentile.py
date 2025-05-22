from odc.algo._percentile import np_percentile
import numpy as np
import pytest


SIZE = (50, 50_000)
NODATA = 101
PECRENTILES = [0.1234567, 0.58274658, 0.6379728]  # make sure there's no tie breaking


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.uint16, np.float32, np.float64]
)
def test_benchmark_np_percentile(benchmark, dtype):
    arr = 100 * np.random.random(SIZE).astype(dtype)
    mask = np.random.random(SIZE) < 0.5
    arr[mask] = NODATA

    def func():
        for p in PECRENTILES:
            _ = np_percentile(arr, p, NODATA)

    benchmark(func)


@pytest.mark.skip("I don't know what this percentile should be")
@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.uint16, np.float32, np.float64]
)
def test_benchmark_percentile(benchmark, dtype):
    from datacube_compute import percentile

    arr = 100 * np.random.random(SIZE).astype(dtype)
    mask = np.random.random(SIZE) < 0.5
    arr[mask] = NODATA

    benchmark(percentile, arr, np.array(PECRENTILES), NODATA)


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.uint16, np.float32, np.float64]
)
def test_percentile(dtype):
    if dtype in (np.float32, np.float64):
        nodata = np.nan
    else:
        nodata = NODATA

    arr = np.array(
        [
            [0, 1, 4, 6, 8, nodata, nodata, nodata, nodata, nodata],
            [3, 5, 6, 8, 9, 11, 15, 28, 31, 50],
        ]
    )

    arr = arr.astype(dtype)

    np.random.shuffle(arr[0, :])
    np.random.shuffle(arr[1, :])
    arr = arr.transpose()

    out = percentile(arr, np.array([0.5, 0.7, 1.0, 0.0]), nodata)
    assert (out[0, :] == np.array([4, 11])).all()
    assert (out[1, :] == np.array([6, 15])).all()
    assert (out[2, :] == np.array([8, 50])).all()
    assert (out[3, :] == np.array([0, 3])).all()


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.uint16, np.float32, np.float64]
)
def test_accuracy(dtype):
    arr = 100 * np.random.random((50, 10, 10)).astype(dtype)
    mask = np.random.random(arr.shape) < 0.5

    if dtype in (np.float32, np.float64):
        nodata = np.nan
    else:
        nodata = NODATA

    arr[mask] = nodata

    out = percentile(arr, np.array(PECRENTILES), nodata)

    for i, p in enumerate(PECRENTILES):
        np.testing.assert_array_equal(out[i, :], np_percentile(arr, p, nodata))


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.uint16, np.float32, np.float64]
)
def test_bad_data(dtype):
    arr = 100 * np.random.random((50, 10, 10)).astype(dtype)
    mask = np.random.random(arr.shape) < 0.9999

    if dtype in (np.float32, np.float64):
        nodata = np.nan
    else:
        nodata = NODATA

    arr[mask] = nodata

    out = percentile(arr, np.array(PECRENTILES), nodata)

    for i, p in enumerate(PECRENTILES):
        np.testing.assert_array_equal(out[i, :], np_percentile(arr, p, nodata))
