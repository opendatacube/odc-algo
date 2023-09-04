from .backend import (
    _geomedian, 
    _geomedian_int16, 
    _geomedian_uint16,
    _percentile_uint16, 
    _percentile_int16,
    _percentile_uint8, 
    _percentile_int8,
    _percentile_f32,
    _percentile_f64,
)

import numpy as np
from collections.abc import Iterable


def geomedian(in_array, **kwargs):

    kwargs.setdefault("num_threads", 1)
    kwargs.setdefault("eps", 1e-6)
    kwargs.setdefault("maxiters", 1000)
    kwargs.setdefault("scale", 1.0)
    kwargs.setdefault("offset", 0.0)

    if len(in_array.shape) != 4:
        raise ValueError(
            f"in_array: expected array to have 4 dimensions in format (y, x, band, time), found {len(in_array.shape)} dimensions."
        )

    if in_array.dtype == np.float32:
        kwargs.setdefault("nodata", np.nan)
        return _geomedian(in_array, kwargs["maxiters"], kwargs["eps"], kwargs["num_threads"], kwargs["scale"], kwargs["offset"])
    elif in_array.dtype == np.int16:
        kwargs.setdefault("nodata", -1)
        return _geomedian_int16(in_array, kwargs["maxiters"], kwargs["eps"], kwargs["num_threads"], kwargs["nodata"], kwargs["scale"], kwargs["offset"])
    elif in_array.dtype == np.uint16:
        kwargs.setdefault("nodata", 0)
        return _geomedian_uint16(in_array, kwargs["maxiters"], kwargs["eps"], kwargs["num_threads"], kwargs["nodata"], kwargs["scale"], kwargs["offset"])
    else:
        raise TypeError(f"in_array: expected dtype to be one of {np.float32}, {np.int16}, {np.uint16}, found {in_array.dtype}.")


def percentile(in_array, percentiles, nodata=None):
    """
    Calculates the percentiles of the input data along the first axis. 

    It accepts an array with shape (t, *other dims) and returns an array with shape 
    (len(percentiles), *other dims) where the first index of the output array correspond to the percentiles.
    e.g. `out[i, :]` corresponds to the ith percentile

    :param in_array: a numpy array

    :param percentiles: A sequence of percentiles or singular percentile in the [0.0, 1.0] range

    :param nodata: The `nodata` value - this must have the same type as in_array.dtype
        and must be provided for integer datatypes. For float types this value is ignored and
        nodata is assumed to be NaN.

    """

    if isinstance(percentiles, Iterable):
        percentiles = np.array(list(percentiles))
    else:
        percentiles = np.array([percentiles])

    shape = in_array.shape
    in_array = in_array.reshape((shape[0], -1))

    if in_array.dtype == np.uint16:
        out_array = _percentile_uint16(in_array, percentiles, nodata)
    elif in_array.dtype == np.int16:
        out_array = _percentile_int16(in_array, percentiles, nodata)
    elif in_array.dtype == np.uint8:
        out_array = _percentile_uint8(in_array, percentiles, nodata)
    elif in_array.dtype == np.int8:
        out_array = _percentile_int8(in_array, percentiles, nodata)
    elif in_array.dtype == np.float32:
        out_array = _percentile_f32(in_array, percentiles)
    elif in_array.dtype == np.float64:
        out_array = _percentile_f64(in_array, percentiles)
    else: 
        raise NotImplementedError

    return out_array.reshape((len(percentiles),) + shape[1:])

__all__ = ("geomedian", "percentile")
