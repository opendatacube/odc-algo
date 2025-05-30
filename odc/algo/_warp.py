# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2025 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from affine import Affine

from odc.geo.warp import rio_warp_affine as warp_affine

from ._numeric import shape_shrink2

if TYPE_CHECKING:
    from ._types import NodataType


def _shrink2(
    xx: np.ndarray,
    resampling: str = "nearest",
    nodata: NodataType | None = None,
    axis: int = 0,
):
    """
    :param xx: Image to shrink
    :param resampling: Resampling strategy to use
    :param nodata: nodata value for missing value fill
    :param axis: Y-axis index, to distinguish Y,X,B (default) vs B,Y,X (axis=1)
    """
    out_shape = shape_shrink2(xx.shape, axis=axis)
    out = np.empty(out_shape, dtype=xx.dtype)

    if xx.ndim == 2:
        warp_affine(
            xx,
            out,
            Affine.scale(2),
            resampling=resampling,
            src_nodata=nodata,
            dst_nodata=nodata,
        )
    elif xx.ndim == 3:
        if axis == 0:
            xx = xx.transpose((2, 0, 1))
            out = out.transpose((2, 0, 1))
        for i in range(xx.shape[0]):
            warp_affine(
                xx[i],
                out[i],
                Affine.scale(2),
                resampling=resampling,
                src_nodata=nodata,
                dst_nodata=nodata,
            )
        if axis == 0:
            out = out.transpose((1, 2, 0))
            assert out_shape == out.shape
    else:
        raise ValueError("Only support Y,X | Y,X,B | B,Y,X inputs")

    return out
