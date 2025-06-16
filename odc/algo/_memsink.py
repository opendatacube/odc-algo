# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2025 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import dask
import dask.array as da
import numpy as np
import xarray as xr
from dask.base import tokenize
from dask.graph_manipulation import bind

from ._dask import _roi_from_chunks, unpack_chunks

if TYPE_CHECKING:
    from collections.abc import Hashable

    from dask.delayed import Delayed
    from distributed import Client

ShapeLike = int | tuple[int, ...]
DtypeLike = str | np.dtype
ROI = slice | tuple[slice, ...]
MaybeROI = ROI | None

_cache: dict[str, np.ndarray] = {}


class Token:
    __slots__ = ["_k"]

    def __init__(self, k: str):
        # print(f"Token.init(<{k}>)@0x{id(self):08X}")
        self._k = k

    def __str__(self) -> str:
        return self._k

    def __bool__(self):
        return len(self._k) > 0

    def release(self):
        if self:
            Cache.pop(self)
            self._k = ""

    def __del__(self):
        # print(f"Token.del(<{self._k}>)@0x{id(self):08X}")
        self.release()

    def __getstate__(self):
        print(f"Token.__getstate__() <{self._k}>@0x{id(self):08X}")
        raise ValueError("Token should not be pickled")

    def __setstate__(self, k):
        print(f"Token.__setstate__(<{k}>)@0x{id(self):08X}")
        raise ValueError("Token should not be pickled")


CacheKey = Token | str


class Cache:
    @staticmethod
    def new(shape: ShapeLike, dtype: DtypeLike) -> Token:
        return Cache.put(np.ndarray(shape, dtype=dtype))

    @staticmethod
    def dask_new(shape: ShapeLike, dtype: DtypeLike, name: str = "") -> Delayed:
        if name == "":
            name = f"mem_array_{dtype!s}"

        name = name + "-" + tokenize(name, shape, dtype)
        return dask.delayed(Cache.new)(shape, dtype, dask_key_name=name)

    @staticmethod
    def collect(
        k: CacheKey, _store_tasks, name: str = ""
    ) -> Delayed | np.ndarray | None:
        # return token if dask objects else underlying memory
        if dask.is_dask_collection(k):
            if name == "":
                name = "mem_array_collect"

            return bind(dask.delayed(lambda k: k), _store_tasks)(
                k, dask_key_name=f"{name}-{k.key}"
            )
        else:
            return Cache.get(k)

    @staticmethod
    def put(x: np.ndarray) -> Token:
        k = uuid.uuid4().hex
        _cache[k] = x
        return Token(k)

    @staticmethod
    def get(k: CacheKey) -> np.ndarray | None:
        return _cache.get(str(k), None)

    @staticmethod
    def pop(k: CacheKey) -> np.ndarray | None:
        return _cache.pop(str(k), None)


class CachedArray:
    def __init__(self, token_or_key: CacheKey):
        self._tk = token_or_key

    @property
    def data(self) -> np.ndarray:
        xx = Cache.get(self._tk)
        if xx is None:
            raise ValueError("Source array is missing from cache")
        return xx

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    def __getitem__(self, key: ROI) -> np.ndarray:
        return self.data[key]

    def __setitem__(self, key, item):
        self.data[key] = item

    @staticmethod
    def new(shape: ShapeLike, dtype: DtypeLike) -> CachedArray:
        return CachedArray(Cache.new(shape, dtype))

    @staticmethod
    def wrap(x: np.ndarray) -> CachedArray:
        return CachedArray(Cache.put(x))

    def release(self) -> np.ndarray | None:
        return Cache.pop(self._tk)


class _YXBTSink:
    def __init__(
        self,
        token_or_key: CacheKey,
        band: int | tuple[slice, slice, slice, slice],
    ):
        if isinstance(band, int):
            band = np.s_[:, :, band, :]

        self._tk = token_or_key
        self._roi = band

    @property
    def data(self):
        xx = Cache.get(self._tk)
        if xx is None:
            return None
        return xx[self._roi]

    def __setitem__(self, key, item):
        assert len(key) == 3
        assert item.ndim == 3

        it, iy, ix = key
        self.data[iy, ix, it] = item.transpose([1, 2, 0])


class _YXTSink:
    def __init__(
        self,
        token_or_key: CacheKey,
    ):
        self._tk = token_or_key

    @property
    def data(self):
        return Cache.get(self._tk)

    def __setitem__(self, key, item):
        assert len(key) == 3
        assert item.ndim == 3

        it, iy, ix = key
        self.data[iy, ix, it] = item.transpose([1, 2, 0])


def store_to_mem(
    xx: da.Array, client: Client, out: np.ndarray | None = None
) -> np.ndarray:
    assert client.scheduler.address.startswith("inproc://")
    token = None
    if out is None:
        sink = dask.delayed(CachedArray.new)(xx.shape, xx.dtype)
    else:
        assert out.shape == xx.shape
        token = Cache.put(out)
        sink = dask.delayed(CachedArray)(str(token))

    try:
        fut = da.store(xx, sink, lock=False, compute=False)
        fut, _sink = client.compute([fut, sink])
        fut.result()
        return _sink.result().data
    finally:
        if token is not None:
            token.release()


def yxbt_sink_to_mem(bands: tuple[da.Array, ...], client: Client) -> np.ndarray:
    assert client.scheduler.address.startswith("inproc://")

    b = bands[0]
    dtype = b.dtype
    nt, ny, nx = b.shape
    nb = len(bands)
    token = Cache.new((ny, nx, nb, nt), dtype)
    sinks = [dask.delayed(_YXBTSink)(token, idx) for idx in range(nb)]
    try:
        fut = da.store(bands, sinks, lock=False, compute=False)
        fut = client.compute(fut)
        fut.result()
        return Cache.get(token)
    finally:
        token.release()


def _chunk_extractor(cache_key: CacheKey, roi: ROI, *deps) -> np.ndarray:
    src = Cache.get(cache_key)
    assert src is not None
    return src[roi]


def _da_from_mem(
    token: Delayed,
    shape: ShapeLike,
    dtype: DtypeLike,
    chunks: tuple[int, ...],
    name: str = "from_mem",
) -> da.Array:
    """
    Construct dask view of some yet to be computed in RAM store.

    :param token: Should evaluate to either Token or string key in to the Cache,
                  which is expected to contain ``numpy`` array of supplied
                  ``shape`` and ``dtype``

    :param shape: Expected shape of the future array

    :param dtype: Expected dtype of the future array

    :param chunks: tuple of integers describing chunk partitioning for output array

    :param name: Dask name

    Gotchas
    =======

    - Output array can not be moved from one worker to another.
      - Works with in-process Client
      - Works with single worker cluster
      - Can work if scheduler is told to schedule this on a single worker

    - Cache life cycle management can be tough. If token evaluates to a
      ``Token`` object then automatic cache cleanup should happen when output
      array is destroyed. If it is just a string, then it's up to caller to
      ensure that there is cleanup and no use after free.

    Returns
    =======
    Dask Array
    """
    if not isinstance(shape, tuple):
        shape = (shape,)

    assert dask.is_dask_collection(token)
    assert len(shape) == len(chunks)

    _chunks = unpack_chunks(chunks, shape)
    _rois = [tuple(_roi_from_chunks(ch)) for ch in _chunks]

    def _roi(idx):
        return tuple(_rois[i][k] for i, k in enumerate(idx))

    shape_in_chunks = tuple(len(ch) for ch in _chunks)

    # chunk delayed and stack back by "shape of grid"
    arr = np.empty(shape_in_chunks, dtype=object)
    for idx in np.ndindex(shape_in_chunks):
        slices = _roi(idx)
        out_shape = tuple((slc.stop - slc.start) for slc in slices)
        d = dask.delayed(_chunk_extractor)(token, slices)
        d = da.from_delayed(d, shape=out_shape, dtype=dtype)

        indices = tuple(slc.start for slc in slices)
        arr_idx = []
        for ax, start in enumerate(indices):
            chunk_size = out_shape[ax]
            _idx = start // chunk_size
            arr_idx.append(_idx)
        arr[tuple(arr_idx)] = d

    darr = da.block(arr.tolist())

    # only to retain the node/task name
    return darr.map_blocks(lambda x: x, name=name)


def da_mem_sink(xx: da.Array, chunks: tuple[int, ...], name="memsink") -> da.Array:
    """
    It's a kind of fancy rechunk for special needs.

    Assumptions
    - Single worker only
    - ``xx`` can fit in RAM of the worker

    Note that every output chunk depends on ALL of input chunks.

    On some Dask worker:
    - Fully evaluate ``xx`` and serialize to RAM
    - Present in RAM view of the result with a different chunking regime

    A common use case would be to load a large collection (>50% of RAM) that
    needs to be processed by some non-Dask code as a whole. A simple
    ``do_stuff(xx.compute())`` would not work as duplicating RAM is not an
    option in that scenario. Normal rechunk might also run out of RAM and
    introduces large memory copy overhead as all input chunks need to be cached
    then re-assembled into a different chunking structure.
    """
    tk = tokenize(xx)

    token = Cache.dask_new(xx.shape, xx.dtype, f"{name}_alloc")
    # Store everything to MEM and only then evaluate to Token
    sink = dask.delayed(CachedArray)(token)
    fut = da.store(xx, sink, lock=False, compute=False)
    sink_name = f"{name}_collect-{tk}"

    token_done = Cache.collect(token, fut, name=sink_name)
    return _da_from_mem(
        token_done, shape=xx.shape, dtype=xx.dtype, chunks=chunks, name=name
    )


def da_yxt_sink(band: da.Array, chunks: tuple[int, int, int], name="yxt") -> da.Array:
    """
    band is in <t,y,x>
    output is <y,x,t>

    eval(band) |> transpose(YXT) |> Store(RAM) |> DaskArray(RAM, chunks)
    """
    tk = tokenize(band, "da_yxt_sink", chunks, name)

    dtype = band.dtype
    nt, ny, nx = band.shape
    shape = (ny, nx, nt)

    token = Cache.dask_new(shape, dtype, f"{name}_alloc")
    sink = dask.delayed(_YXTSink)(token)
    fut = da.store(band, sink, lock=False, compute=False)
    sink_name = f"{name}_collect-{tk}"
    token_done = Cache.collect(token, fut, name=sink_name)

    return _da_from_mem(token_done, shape=shape, dtype=dtype, chunks=chunks, name=name)


def da_yxbt_sink(
    bands: tuple[da.Array, ...], chunks: tuple[int, ...], dtype=None, name="yxbt"
) -> da.Array:
    """
    each band is in <t,y,x>
    output is <y,x,b,t>

    eval(bands) |> transpose(YXBT) |> Store(RAM) |> DaskArray(RAM, chunks)
    """
    tk = tokenize(*bands, chunks, name)

    b = bands[0]
    if dtype is None:
        dtype = b.dtype
    nt, ny, nx = b.shape
    nb = len(bands)
    shape = (ny, nx, nb, nt)

    token = Cache.dask_new(shape, dtype, f"{name}_alloc")
    sinks = [dask.delayed(_YXBTSink)(token, idx) for idx in range(nb)]
    fut = da.store(bands, sinks, lock=False, compute=False)
    sink_name = f"{name}_collect-{tk}"

    token_done = Cache.collect(token, fut, name=sink_name)

    return _da_from_mem(token_done, shape=shape, dtype=dtype, chunks=chunks, name=name)


def yxbt_sink(
    ds: xr.Dataset, chunks: tuple[int, int, int, int], dtype=None, name="yxbt"
) -> xr.DataArray:
    """
    Given a Dask dataset with several bands and ``T,Y,X`` axis order on input,
    turn that into a Dask DataArray with axis order being ``Y, X, Band, T``.

    The way this function work is
    - Evaluate all input data before making any output chunk available for further processing
    - For each input block store it into appropriate location in RAM.
    - Expose in RAM store as Dask Array with requested chunking regime

    This is used for Geomedian computation mostly, for GM chunks need to be ``(ny, nx, -1,-1)``.

    :param ds: Dataset with Dask based arrays ``T,Y,X`` axis order
    :param chunks: Chunk size for output array, example: ``(100, 100, -1, -1)``
    :param dtype: dtype of Array to sink to
    :param name: A name given to generate memory cache token
       WARNINGS: if left as default with >= 2 DataArrays with same dtype and shape
       will cause "broken" memory

    Gotchas
    =======
    - Output array can not be moved from one worker to another.
      - Works with in-process Client
      - Works with single worker cluster
      - Can work if scheduler is told to schedule this on a single worker


    Returns
    =======
    xarray DataArray backed by Dask array.
    """
    b0, *_ = ds.data_vars.values()
    data = da_yxbt_sink(
        tuple(dv.data for dv in ds.data_vars.values()), chunks, dtype=dtype, name=name
    )
    attrs = dict(b0.attrs)
    dims = b0.dims[1:] + ("band", b0.dims[0])

    coords: dict[Hashable, Any] = dict(ds.coords.items())
    coords["band"] = list(ds.data_vars)

    return xr.DataArray(data=data, dims=dims, coords=coords, attrs=attrs)


def yxt_sink(
    band: xr.DataArray, chunks: tuple[int, int, int], name="yxt"
) -> xr.DataArray:
    """
    Load ``T,Y,X` dataset into RAM with transpose to ``Y,X,T``, then present
    that as Dask array with specified chunking.

    :param band:
       Dask backed :class:`xr.DataArray` data in ``T,Y,X`` order
    :param chunks:
       Desired output chunk size in output order ``Y,X,T``
    :param name:
       A name given to generate memory cache token
       WARNINGS: if left as default with >= 2 DataArrays with same dtype and shape
       will cause "broken" memory
    :return:
       Dask backed :class:`xr.DataArray` with requested chunks and ``Y,X,T`` axis order
    """
    data = da_yxt_sink(band.data, chunks=chunks, name=name)
    dims = band.dims[1:] + (band.dims[0],)
    return xr.DataArray(data=data, dims=dims, coords=band.coords, attrs=band.attrs)
