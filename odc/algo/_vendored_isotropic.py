# This file was vendored from scikit-image
# https://github.com/scikit-image/scikit-image/blob/47099c03b4c45a882c5fbd48e99b5c15db554ffe/src/skimage/morphology/isotropic.py
#
# Copyright: 2009-2022 the scikit-image team
# SPDX-License-Identifier: BSD-3-Clause
# License: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the University nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# .
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Binary morphological operations
"""

import numpy as np
from scipy import ndimage as ndi


def isotropic_erosion(image, radius, out=None, spacing=None):
    """Return binary morphological erosion of an image.

    Compared to the more general :func:`skimage.morphology.erosion`, this
    function only supports binary inputs and circular footprints.
    However, it performs typically faster for large (circular) footprints.
    This works by applying a threshold to the exact Euclidean distance map
    of the image [1]_, [2]_.
    The implementation is based on: func:`scipy.ndimage.distance_transform_edt`.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius of the footprint used for the operation.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None,
        a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension.
        If a sequence, must be of length equal to the input's dimension (number of axes).
        If a single number, this value is used for all axes.
        If not specified, a grid spacing of unity is implied.

    Returns
    -------
    eroded : ndarray of bool
        The result of the morphological erosion taking values in
        ``[False, True]``.

    References
    ----------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`

    Examples
    --------
    Erosion shrinks bright regions

    >>> import numpy as np
    >>> import skimage as ski
    >>> image = np.array([[0, 0, 1, 0, 0],
    ...                   [0, 1, 1, 1, 0],
    ...                   [0, 1, 1, 1, 0],
    ...                   [0, 1, 1, 1, 0],
    ...                   [0, 0, 0, 0, 0]], dtype=bool)
    >>> result = ski.morphology.isotropic_erosion(image, radius=1)
    >>> result.view(np.uint8)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)
    """

    dist = ndi.distance_transform_edt(image, sampling=spacing)
    return np.greater(dist, radius, out=out)


def isotropic_dilation(image, radius, out=None, spacing=None):
    """Return binary morphological dilation of an image.

    Compared to the more general :func:`skimage.morphology.dilation`, this
    function only supports binary inputs and circular footprints.
    However, it performs typically faster for large (circular) footprints.
    This works by applying a threshold to the exact Euclidean distance map
    of the inverted image [1]_, [2]_.
    The implementation is based on: func:`scipy.ndimage.distance_transform_edt`.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius of the footprint used for the operation.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension.
        If a sequence, must be of length equal to the input's dimension (number of axes).
        If a single number, this value is used for all axes.
        If not specified, a grid spacing of unity is implied.

    Returns
    -------
    dilated : ndarray of bool
        The result of the morphological dilation with values in
        ``[False, True]``.

    References
    ----------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`

    Examples
    --------
    Dilation enlarges bright regions

    >>> import numpy as np
    >>> import skimage as ski
    >>> image = np.array([[0, 0, 0, 0, 0],
    ...                   [0, 0, 0, 0, 0],
    ...                   [0, 0, 1, 0, 0],
    ...                   [0, 0, 1, 1, 0],
    ...                   [0, 0, 0, 0, 0]], dtype=bool)
    >>> result = ski.morphology.isotropic_dilation(image, radius=1)
    >>> result.view(np.uint8)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 1],
           [0, 0, 1, 1, 0]], dtype=uint8)
    """

    dist = ndi.distance_transform_edt(np.logical_not(image), sampling=spacing)
    return np.less_equal(dist, radius, out=out)


def isotropic_opening(image, radius, out=None, spacing=None):
    """Return binary morphological opening of an image.

    Compared to the more general :func:`skimage.morphology.opening`, this
    function only supports binary inputs and circular footprints.
    However, it performs typically faster for large (circular) footprints.
    This works by thresholding the exact Euclidean distance map [1]_, [2]_.
    The implementation is based on: func:`scipy.ndimage.distance_transform_edt`.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius of the footprint used for the operation.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None
        is passed, a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension.
        If a sequence, must be of length equal to the input's dimension (number of axes).
        If a single number, this value is used for all axes.
        If not specified, a grid spacing of unity is implied.

    Returns
    -------
    opened : ndarray of bool
        The result of the morphological opening.

    References
    ----------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`

    Examples
    --------
    Remove connection between two bright regions

    >>> import numpy as np
    >>> import skimage as ski
    >>> image = np.array([[1, 0, 0, 0, 1],
    ...                   [1, 1, 0, 1, 1],
    ...                   [1, 1, 1, 1, 1],
    ...                   [1, 1, 0, 1, 1],
    ...                   [1, 0, 0, 0, 1]], dtype=bool)
    >>> result = ski.morphology.isotropic_opening(image, radius=1)
    >>> result.view(np.uint8)
    array([[1, 0, 0, 0, 1],
           [1, 1, 0, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 0, 1, 1],
           [1, 0, 0, 0, 1]], dtype=uint8)
    """

    eroded = isotropic_erosion(image, radius, out=out, spacing=spacing)
    return isotropic_dilation(eroded, radius, out=out, spacing=spacing)


def isotropic_closing(image, radius, out=None, spacing=None):
    """Return binary morphological closing of an image.

    Compared to the more general :func:`skimage.morphology.closing`, this
    function only supports binary inputs and circular footprints.
    However, it performs typically faster for large (circular) footprints.
    This works by thresholding the exact Euclidean distance map [1]_, [2]_.
    The implementation is based on: func:`scipy.ndimage.distance_transform_edt`.

    Parameters
    ----------
    image : ndarray
        Binary input image.
    radius : float
        The radius of the footprint used for the operation.
    out : ndarray of bool, optional
        The array to store the result of the morphology. If None,
        is passed, a new array will be allocated.
    spacing : float, or sequence of float, optional
        Spacing of elements along each dimension.
        If a sequence, must be of length equal to the input's dimension (number of axes).
        If a single number, this value is used for all axes.
        If not specified, a grid spacing of unity is implied.

    Returns
    -------
    closed : ndarray of bool
        The result of the morphological closing.

    References
    ----------
    .. [1] Cuisenaire, O. and Macq, B., "Fast Euclidean morphological operators
        using local distance transformation by propagation, and applications,"
        Image Processing And Its Applications, 1999. Seventh International
        Conference on (Conf. Publ. No. 465), 1999, pp. 856-860 vol.2.
        :DOI:`10.1049/cp:19990446`

    .. [2] Ingemar Ragnemalm, Fast erosion and dilation by contour processing
        and thresholding of distance maps, Pattern Recognition Letters,
        Volume 13, Issue 3, 1992, Pages 161-166.
        :DOI:`10.1016/0167-8655(92)90055-5`

    Examples
    --------
    Close gap between two bright lines

    >>> import numpy as np
    >>> import skimage as ski
    >>> image = np.array([[0, 0, 0, 0, 0],
    ...                   [0, 0, 0, 0, 0],
    ...                   [1, 1, 0, 1, 1],
    ...                   [0, 0, 0, 0, 0],
    ...                   [0, 0, 0, 0, 0]], dtype=bool)
    >>> result = ski.morphology.isotropic_closing(image, radius=1)
    >>> result.view(np.uint8)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [1, 1, 0, 1, 1],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)
    """

    dilated = isotropic_dilation(image, radius, out=out, spacing=spacing)
    return isotropic_erosion(dilated, radius, out=out, spacing=spacing)