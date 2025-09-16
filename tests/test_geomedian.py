# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2025 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import dask
import dask.array as da
import numpy as np
from xarray import DataArray, Dataset

from odc.algo import geomedian_with_mads, wait_for_future


def test_geomedian_yxbt():
    NT, NY, NX = 3, 1000, 2000
    NB = 5
    rng = np.random.default_rng()
    np_yxbt = rng.random(dtype=da.float32, size=(NY, NX, NB, NT))
    yxbt = da.from_array(np_yxbt, chunks=(100, 101, -1, -1))

    xcoords = list(range(0, NX * 5, 5))
    ycoords = list(range(0, NY * 5, 5))
    tcoords = [
        np.datetime64("2023-04-07"),
        np.datetime64("2023-04-08"),
        np.datetime64("2023-04-09"),
    ]
    bcoords = ["b1", "b2", "b3", "b4", "b5"]
    src = DataArray(
        yxbt,
        coords={"x": xcoords, "y": ycoords, "time": tcoords, "band": bcoords},
        dims=["y", "x", "band", "time"],
    )
    result = geomedian_with_mads(src)
    assert dask.is_dask_collection(result)
    computed_result = result.compute()
    assert not dask.is_dask_collection(computed_result)
    assert "band" not in computed_result.data_vars
    assert "b1" in computed_result.data_vars
    assert "bcmad" in computed_result.data_vars
    assert "band" not in computed_result.dims
    assert "x" in computed_result.dims


def test_geomedian_dataset():
    NT, NY, NX = 3, 1000, 2000

    xcoords = list(range(0, NX * 5, 5))
    ycoords = list(range(0, NY * 5, 5))
    tcoords = [
        np.datetime64("2023-04-07"),
        np.datetime64("2023-04-08"),
        np.datetime64("2023-04-09"),
    ]

    # Could specify a seed here to guarantee same random input every test, but doesn't really matter for this test.
    rng = np.random.default_rng()
    b1 = DataArray(
        da.from_array(
            rng.random(dtype=da.float32, size=(NT, NY, NX)), chunks=(1, 100, 101)
        ),
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
        dims=["time", "y", "x"],
    )
    b2 = DataArray(
        da.from_array(
            rng.random(dtype=da.float32, size=(NT, NY, NX)), chunks=(1, 100, 101)
        ),
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
        dims=["time", "y", "x"],
    )
    b3 = DataArray(
        da.from_array(
            rng.random(dtype=da.float32, size=(NT, NY, NX)), chunks=(1, 100, 101)
        ),
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
        dims=["time", "y", "x"],
    )
    b4 = DataArray(
        da.from_array(
            rng.random(dtype=da.float32, size=(NT, NY, NX)), chunks=(1, 100, 101)
        ),
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
        dims=["time", "y", "x"],
    )
    b5 = DataArray(
        da.from_array(
            rng.random(dtype=da.float32, size=(NT, NY, NX)), chunks=(1, 100, 101)
        ),
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
        dims=["time", "y", "x"],
    )

    src = Dataset(
        data_vars={
            "b1": b1,
            "b2": b2,
            "b3": b3,
            "b4": b4,
            "b5": b5,
        },
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
    )
    result = geomedian_with_mads(src)
    assert dask.is_dask_collection(result)
    computed_result = result.compute()
    assert not dask.is_dask_collection(computed_result)
    assert "band" not in computed_result.data_vars
    assert "b1" in computed_result.data_vars
    assert "bcmad" in computed_result.data_vars
    assert "band" not in computed_result.dims
    assert "x" in computed_result.dims


# @pytest.mark.skip
def test_geomedian_mem():
    client = dask.distributed.Client(
        n_workers=1, threads_per_worker=1, memory_limit=9663676416, processes=False
    )
    # Sleep long enough to start dask console
    # time.sleep(8)
    NT, NY, NX = 7, 3200, 3200
    work_chunks = (400, 400)

    xcoords = list(range(0, NX * 5, 5))
    ycoords = list(range(0, NY * 5, 5))
    tcoords = [
        np.datetime64("2023-04-07"),
        np.datetime64("2023-04-08"),
        np.datetime64("2023-04-09"),
        np.datetime64("2023-04-10"),
        np.datetime64("2023-04-11"),
        np.datetime64("2023-04-12"),
        np.datetime64("2023-04-13"),
        np.datetime64("2023-04-14"),
        np.datetime64("2023-04-15"),
        np.datetime64("2023-04-16"),
        np.datetime64("2023-04-17"),
        np.datetime64("2023-04-18"),
        np.datetime64("2023-04-19"),
        np.datetime64("2023-04-20"),
    ][:NT]

    rng = np.random.default_rng()
    b1 = DataArray(
        da.from_array(
            rng.random(dtype=da.float32, size=(NT, NY, NX)), chunks=(1, 400, 400)
        ),
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
        dims=["time", "y", "x"],
    )
    b2 = DataArray(
        da.from_array(
            rng.random(dtype=da.float32, size=(NT, NY, NX)), chunks=(1, 400, 400)
        ),
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
        dims=["time", "y", "x"],
    )
    b3 = DataArray(
        da.from_array(
            rng.random(dtype=da.float32, size=(NT, NY, NX)), chunks=(1, 400, 400)
        ),
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
        dims=["time", "y", "x"],
    )
    b4 = DataArray(
        da.from_array(
            rng.random(dtype=da.float32, size=(NT, NY, NX)), chunks=(1, 400, 400)
        ),
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
        dims=["time", "y", "x"],
    )
    b5 = DataArray(
        da.from_array(
            rng.random(dtype=da.float32, size=(NT, NY, NX)), chunks=(1, 400, 400)
        ),
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
        dims=["time", "y", "x"],
    )
    b6 = DataArray(
        da.from_array(
            rng.random(dtype=da.float32, size=(NT, NY, NX)), chunks=(1, 400, 400)
        ),
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
        dims=["time", "y", "x"],
    )

    src = Dataset(
        data_vars={
            "b1": b1,
            "b2": b2,
            "b3": b3,
            "b4": b4,
            "b5": b5,
            "b6": b6,
        },
        coords={"x": xcoords, "y": ycoords, "time": tcoords},
    )
    result = geomedian_with_mads(
        src,
        max_iters=2000,
        num_threads=1,
        out_chunks=(-1, -1, -1),
        work_chunks=work_chunks,
    )
    assert dask.is_dask_collection(result)
    running_result = client.compute(result)
    for _ in wait_for_future(running_result, 100):
        pass
    complete_result = running_result.result()
    assert not dask.is_dask_collection(complete_result)
    assert "band" not in complete_result.data_vars
    assert "b1" in complete_result.data_vars
    assert "bcmad" in complete_result.data_vars
    assert "band" not in complete_result.dims
    assert "x" in complete_result.dims
