# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2025 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""Various Algorithmic Helpers"""

from importlib.metadata import version

from ._broadcast import pool_broadcast
from ._dask import (
    chunked_persist,
    chunked_persist_da,
    chunked_persist_ds,
    randomize,
    reshape_yxbt,
    wait_for_future,
)
from ._dask_stream import dask_compute_stream, seq_to_bags
from ._geomedian import (
    geomedian_with_mads,
    int_geomedian,
    int_geomedian_np,
    reshape_for_geomedian,
    xr_geomedian,
)
from ._masking import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    choose_first_valid,
    enum_to_bool,
    erase_bad,
    fmask_to_bool,
    from_float,
    from_float_np,
    gap_fill,
    keep_good_np,
    keep_good_only,
    mask_cleanup,
    mask_cleanup_np,
    to_f32,
    to_f32_np,
    to_float,
    to_float_np,
)
from ._memsink import (
    da_mem_sink,
    da_yxbt_sink,
    da_yxt_sink,
    store_to_mem,
    yxbt_sink,
    yxbt_sink_to_mem,
    yxt_sink,
)
from ._numexpr import apply_numexpr, expr_eval, safe_div
from ._percentile import xr_quantile
from ._rgba import colorize, is_rgb, to_rgba, to_rgba_np
from ._tiff import save_cog

__version__ = version("odc-algo")

__all__ = (
    "__version__",
    "apply_numexpr",
    "binary_closing",
    "binary_dilation",
    "binary_erosion",
    "binary_opening",
    "choose_first_valid",
    "chunked_persist",
    "chunked_persist_da",
    "chunked_persist_ds",
    "colorize",
    "da_mem_sink",
    "da_yxbt_sink",
    "da_yxt_sink",
    "dask_compute_stream",
    "enum_to_bool",
    "erase_bad",
    "expr_eval",
    "fmask_to_bool",
    "from_float",
    "from_float_np",
    "gap_fill",
    "geomedian_with_mads",
    "int_geomedian",
    "int_geomedian_np",
    "is_rgb",
    "keep_good_np",
    "keep_good_only",
    "mask_cleanup",
    "mask_cleanup_np",
    "pool_broadcast",
    "randomize",
    "reshape_for_geomedian",
    "reshape_yxbt",
    "safe_div",
    "save_cog",
    "seq_to_bags",
    "store_to_mem",
    "to_f32",
    "to_f32_np",
    "to_float",
    "to_float_np",
    "to_rgba",
    "to_rgba_np",
    "wait_for_future",
    "xr_geomedian",
    "xr_quantile",
    "yxbt_sink",
    "yxbt_sink_to_mem",
    "yxt_sink",
)
