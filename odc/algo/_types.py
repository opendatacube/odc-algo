# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2025 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np

NumpyIndex1 = int | slice
NumpyIndex2 = tuple[NumpyIndex1, NumpyIndex1]
NumpyIndex = tuple[NumpyIndex1, ...]
NodataType = int | float
ShapeLike = int | tuple[int, ...]
DtypeLike = str | np.dtype
