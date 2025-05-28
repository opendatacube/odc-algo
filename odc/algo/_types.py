# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2025 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from typing import Tuple, Union

NumpyIndex1 = Union[int, slice]
NumpyIndex2 = Tuple[NumpyIndex1, NumpyIndex1]
NumpyIndex = Tuple[NumpyIndex1, ...]
NodataType = Union[int, float]
ShapeLike = Union[int, Tuple[int, ...]]
DtypeLike = Union[str, np.dtype]
