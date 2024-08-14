import time
import numpy as np
import pytest

pytest.skip("No hdstats atm", allow_module_level=True)


from datacube_compute import geomedian
from hdstats import nangeomedian_pcm, smad_pcm, emad_pcm, bcmad_pcm


@pytest.fixture
def arr():
    xx = np.random.random((100, 100, 10, 50)).astype(np.float32)
    mask = np.random.random((100, 100, 10, 50)) < 0.01
    xx[mask] = np.nan
    return xx


@pytest.fixture
def kwargs():
    return {"maxiters": 1000, "eps": 1e-5, "num_threads": 1}


@pytest.fixture
def kwargs_parallel():
    return {"maxiters": 1000, "eps": 1e-5, "num_threads": 4}


def test_benchmark_rust(benchmark, arr, kwargs):
    benchmark(geomedian, arr, **kwargs)


def test_benchmark_hdstats_opensource(benchmark, arr, kwargs):
    def func():
        gm = nangeomedian_pcm(arr, nocheck=True, **kwargs)
        smad = smad_pcm(arr, gm, **kwargs)
        bcmad = bcmad_pcm(arr, gm, **kwargs)
        emad = emad_pcm(arr, gm, **kwargs)

    benchmark(func)


def test_benchmark_parallel_rust(benchmark, arr, kwargs_parallel):
    benchmark(geomedian, arr, **kwargs_parallel)


def test_benchmark_hdstats_opensource_parallel(benchmark, arr, kwargs_parallel):
    def func():
        gm = nangeomedian_pcm(arr, nocheck=True, **kwargs_parallel)
        smad = smad_pcm(arr, gm, **kwargs_parallel)
        bcmad = bcmad_pcm(arr, gm, **kwargs_parallel)
        emad = emad_pcm(arr, gm, **kwargs_parallel)

    benchmark(func)


def test_benchmark_hdstats_nangeomedian_pcm_as_delivered(
    benchmark, arr, kwargs_parallel
):
    from hdstats_ad import nangeomedian_pcm, smad_pcm, emad_pcm

    kwargs_parallel_mads = kwargs_parallel.copy()
    del kwargs_parallel_mads["maxiters"]
    del kwargs_parallel_mads["eps"]

    def func():
        gm = nangeomedian_pcm(arr, **kwargs_parallel)
        smad = smad_pcm(arr, gm, **kwargs_parallel_mads)
        bcmad = bcmad_pcm(arr, gm, **kwargs_parallel)
        emad = emad_pcm(arr, gm, **kwargs_parallel_mads)

    benchmark(func)


def test_accuracy(arr, kwargs):
    gm_1, mads_1 = geomedian(arr, **kwargs)
    gm_2 = nangeomedian_pcm(arr, **kwargs)

    # use rust geomedian to calculate mads
    smad = smad_pcm(arr, gm_1, **kwargs)
    bcmad = bcmad_pcm(arr, gm_1, **kwargs)
    emad = emad_pcm(arr, gm_1, **kwargs)

    dists = np.sqrt(((gm_1 - gm_2) ** 2).sum(axis=-1))

    assert (dists < kwargs["eps"]).all()

    assert np.allclose(emad, mads_1[:, :, 0])
    assert np.allclose(smad, mads_1[:, :, 1])
    assert np.allclose(bcmad, mads_1[:, :, 2])


def test_novalid_measurements(arr, kwargs):
    arr_bad = arr.copy()
    arr_bad[1, 2, :, :] = np.nan

    gm_1, mads_1 = geomedian(arr, **kwargs)
    gm_2, mads_2 = geomedian(arr_bad, **kwargs)

    for arr_1, arr_2 in [(gm_1, gm_2), (mads_1, mads_2)]:
        assert (arr_1[:1, :2, :] == arr_2[:1, :2, :]).all()
        assert (arr_1[2:, 3:, :] == arr_2[2:, 3:, :]).all()
        assert (arr_1[1, :2, :] == arr_2[1, :2, :]).all()
        assert (arr_1[1, 3:, :] == arr_2[1, 3:, :]).all()
        assert (arr_1[1, :2, :] == arr_2[1, :2, :]).all()
        assert (arr_1[:1, 2, :] == arr_2[:1, 2, :]).all()
        assert (arr_1[2:, 2, :] == arr_2[2:, 2, :]).all()
        assert np.isnan(arr_2[1, 2, :]).all()


def test_transform(arr, kwargs):
    args_1 = kwargs.copy()

    args_1["scale"] = np.float32(11.0)
    args_1["offset"] = np.float32(3.0)

    arr_1 = args_1["scale"] * arr + args_1["offset"]

    # passing the transform params should have the same result
    gm_1, _ = geomedian(arr, **args_1)
    gm_2, _ = geomedian(arr_1, **kwargs)

    inv_scale = np.float32(1.0) / args_1["scale"]
    gm_2 = inv_scale * (gm_2 - args_1["offset"])

    assert (gm_1 == gm_2).all()
