"""
Helper methods for Geometric Median computation.
"""
import dask
import dask.array as da
import functools
import numpy as np
import xarray as xr
from typing import Optional, Tuple, Union
from ._geomedian_impl import geomedian



def dataset_block_processor(
        input: xr.Dataset,
        nodata=None,
        scale=1,
        offset=0,
        eps=1e-6,
        maxiters=1000,
        num_threads=1,
    ):

    array = input.to_array(dim="band").transpose("y", "x", "band", "time")
    nodata = array.attrs.get("nodata", None)

    gm_data, mads = geomedian(array.data,
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
    # Add an empty axis so we can concatenate
    #count = count[..., np.newaxis]
    result["count"] = xr.DataArray(data=count, dims=dims[:2], coords=result.coords)

    # TODO: Work out if the following is required
#    for dv in result.data_vars.values():
#        dv.attrs.update(input.attrs)
    
    return result


def _gm_mads_compute_f32(
    yxbt,
    nodata=None,
    scale=1,
    offset=0,
    eps=None,
    maxiters=1000,
    num_threads=1,
):
    """
    output axis order is:

      y, x, band

    When extra stats are compute they are returned in the following order:
    [*bands, emad, smad, bcmad, count]

    note that when supplying non-float input, it is scaled according to scale/offset/nodata parameters,
    output is however returned in that scaled range.
    """

    gm, mads_array = geomedian(
        yxbt,
        nodata=nodata,
        num_threads=num_threads,
        eps=eps,
        maxiters=maxiters,
        scale=scale,
        offset=offset,
    )
    # Compute the count in Python/NumPy
    nbads = np.isnan(yxbt).sum(axis=2, dtype="bool").sum(axis=2, dtype="uint16")
    count = yxbt.dtype.type(yxbt.shape[-1]) - nbads
    # Add an empty axis so we can concatenate
    count = count[..., np.newaxis]

    # Jam all the arrays together. Which is weird, because we then proceed to pull them
    # back apart again.
    return np.concatenate([gm, mads_array, count], axis=2)


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
    if not compute_mads:
        raise ValueError("compute_mads must be set to True")
    if not compute_count:
        raise ValueError("compute_count must be set to True")

    if not dask.is_dask_collection(src):
        raise ValueError("This method only works on Dask inputs")

#    if isinstance(src, xr.DataArray):
#        yxbt = src
#    else:
#        ny, nx = work_chunks
#        if reshape_strategy == "mem":
#            yxbt = yxbt_sink(src, (ny, nx, -1, -1))
#        elif reshape_strategy == "yxbt":
#            yxbt = reshape_yxbt(src, yx_chunks=(ny, nx))
#        else:
#            raise ValueError(
#                f"Reshape strategy '{reshape_strategy}' not understood use one of: mem or yxbt"
#            )

    ny, nx = work_chunks
    chunked = src.chunk({"y": ny, "x": nx, "time": -1})

    # TODO: I don't think this is needed, since we *just* specified it
#    ny, nx, nb, nt = yxbt.shape
#    nodata = chunked.attrs.get("nodata", None)
#    assert yxbt.chunks is not None
#    if yxbt.data.numblocks[2:4] != (1, 1):
#        raise ValueError("There should be one dask block along time and band dimension")

#    chunks = (*yxbt.chunks[:2], (nb + n_extras,))

    # Check the dtype of the first data variable
    is_float = next(iter(src.dtypes.values())) == "f"

    if eps is None:
        eps = 1e-4 if is_float else 0.1 * scale

#    op = functools.partial(
#        _gm_mads_compute_f32,
#        nodata=nodata,
#        scale=scale,
#        offset=offset,
#        eps=eps,
#        maxiters=maxiters,
#        num_threads=num_threads,
#    )

    _gm_with_mads = chunked.map_blocks(
            dataset_block_processor,
            kwargs=dict(
                   scale=scale,
                   offset=offset,
                   eps=eps,
                   maxiters=maxiters,
                   num_threads=num_threads,
                )
    )

#    _gm = da.map_blocks(
#        op, yxbt.data, dtype=yxbt.dtype, drop_axis=3, chunks=chunks, name="geomedian"
#    )
# TODO: Check, but this seems bogus now, we haven't gone via a single array.
    # if out_chunks is not None:
    #     _gm_with_mads = _gm_with_mads.chunk(out_chunks)

    return _gm_with_mads
