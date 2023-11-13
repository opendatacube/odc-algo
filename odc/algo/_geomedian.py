"""
Helper methods for Geometric Median computation.
"""
from typing import Optional, Tuple, Union

import dask
import numpy as np
import xarray as xr

from ._geomedian_impl import geomedian


def geomedian_block_processor(
    input: xr.Dataset,
    nodata=None,
    scale=1,
    offset=0,
    eps=1e-6,
    maxiters=1000,
    num_threads=1,
):
    # Le sigh, more issues with "spec" vs "time"
    if "time" in input.dims:
        array = input.to_array(dim="band").transpose("y", "x", "band", "time")
    else:
        array = input.to_array(dim="band").transpose("y", "x", "band", "spec")

    if nodata is None:
        nodata = input.attrs.get("nodata", None)

    if nodata is None:
        # Grab the nodata value from our input array
        nodata_vals = set(
            dv.attrs.get("nodata", None) for dv in input.data_vars.values()
        )
        if len(nodata_vals) > 1:
            raise ValueError("I can't handle more than 1 nodata value!", nodata_vals)
        elif len(nodata_vals) == 1:
            nodata = nodata_vals.pop()
        else:
            # Not sure if this is a sensible default.
            nodata = float("nan")

    gm_data, mads = geomedian(
        array.data,
        nodata=nodata,
        num_threads=num_threads,
        eps=eps,
        maxiters=maxiters,
        scale=scale,
        offset=offset,
    )

    dims = ("y", "x", "band")
    coords = {k: array.coords[k] for k in dims}
    result = xr.DataArray(
        data=gm_data, dims=dims, coords=coords, attrs=array.attrs
    ).to_dataset("band")

    emad = mads[:, :, 0]
    smad = mads[:, :, 1]
    bcmad = mads[:, :, 2]

    # TODO: Work out if the following is required
    # if not is_float:
    #     emad = emad * (1 / scale)

    result["emad"] = xr.DataArray(data=emad, dims=dims[:2], coords=result.coords)
    result["smad"] = xr.DataArray(data=smad, dims=dims[:2], coords=result.coords)
    result["bcmad"] = xr.DataArray(data=bcmad, dims=dims[:2], coords=result.coords)

    # Compute the count in Python/NumPy
    nbads = np.isnan(array.data).sum(axis=2, dtype="bool").sum(axis=2, dtype="uint16")
    count = array.dtype.type(array.shape[-1]) - nbads
    result["count"] = xr.DataArray(
        data=count, dims=dims[:2], coords=result.coords
    ).astype("uint16")

    # TODO: Work out if the following is required
    #    for dv in result.data_vars.values():
    #        dv.attrs.update(input.attrs)

    return result


def geomedian_with_mads(
    src: Union[xr.Dataset, xr.DataArray],
    compute_mads: bool = True,
    compute_count: bool = True,
    out_chunks: Optional[Tuple[int, int, int]] = None,
    reshape_strategy: str = "mem",
    scale: float = 1.0,
    offset: float = 0.0,
    eps: Optional[float] = None,
    maxiters: int = 1000,
    num_threads: int = 1,
    work_chunks: Tuple[int, int] = (100, 100),
) -> xr.Dataset:
    """
    Compute Geomedian on Dask backed Dataset.

    NOTE: Default configuration of this code assumes that entire input can be
    loaded in to RAM on the Dask worker. It also assumes that there is only one
    worker in the cluster, or that entire task will get scheduled on one single
    worker only. See ``reshape_strategy`` parameter.

    :param src: xr.Dataset or a single array in YXBT order, bands can be either
                float or integer with `nodata` values to indicate gaps in data.

    :param compute_mads: Whether to compute smad,emad,bcmad statistics

    :param compute_count: Whether to compute count statistic (number of
                          contributing observations per output pixels)

    :param out_chunks: Advanced option, allows to rechunk output internally,
                       order is ``(ny, nx, nband)``

    :param reshape_strategy: One of ``mem`` (default) or ``yxbt``. This is only
    applicable when supplying Dataset object. It controls how Dataset is
    reshaped into DataArray in the format expected by Geomedian code. If you
    have enough RAM and use single-worker Dask cluster, then use ``mem``, it
    should be the most efficient. If there is not enough RAM to load entire
    input you can try ``yxbt`` mode, but you might still run out of RAM anyway.
    If using multi-worker Dask cluster you have to use ``yxbt`` strategy.

    :param scale, offset: Only used when input contains integer values, actual
                          Geomedian will run on scaled values
                          ``scale*X+offset``. Only affects internal
                          computation, final result is scaled back to the
                          original value range.

    :param eps: Termination criteria passed on to geomedian algorithm

    :param maxiters: Maximum number of iterations done per output pixel

    :param num_threads: Configure internal concurrency of the Geomedian
                        computation. Default is 1 as we assume that Dask will
                        run a bunch of those concurrently.

    :param work_chunks: Default is ``(100, 100)``, only applicable when input
                        is Dataset.
    """
    # Validate Arguments
    if not compute_mads:
        raise ValueError("compute_mads must be set to True")
    if not compute_count:
        raise ValueError("compute_count must be set to True")
    if not dask.is_dask_collection(src):
        raise ValueError("This method only works on Dask inputs")

    ny, nx = work_chunks
    # TODO: This is kind of horrible, I don't know why odc-algo is replacing the
    # time dimension with the spec dimension when going through load_with_native_transform
    # and in particular, group_by_nothing()
    # So work around poorly for now.
    if "spec" in src.dims:
        chunked = src.chunk({"y": ny, "x": nx, "spec": -1})
    else:
        chunked = src.chunk({"y": ny, "x": nx, "time": -1})

    # Check the dtype of the first data variable
    is_float = next(iter(src.dtypes.values())) == "f"

    if eps is None:
        eps = 1e-4 if is_float else 0.1 * scale

    _gm_with_mads = chunked.map_blocks(
        geomedian_block_processor,
        kwargs=dict(
            scale=scale,
            offset=offset,
            eps=eps,
            maxiters=maxiters,
            num_threads=num_threads,
        ),
    )

    return _gm_with_mads
