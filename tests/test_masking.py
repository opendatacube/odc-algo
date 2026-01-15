# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2026 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from odc.algo._masking import (
    _disk,
    _enum_to_mask_numexpr,
    _fuse_mean_np,
    _gap_fill_np,
    _get_enum_values,
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    enum_to_bool,
    fmask_to_bool,
    gap_fill,
    mask_cleanup,
    mask_cleanup_np,
)


def test_gap_fill():
    a = np.zeros((5,), dtype="uint8")
    b = np.empty_like(a)
    b[:] = 33

    a[0] = 11
    ab = _gap_fill_np(a, b, 0)
    assert ab.dtype == a.dtype
    assert ab.tolist() == [11, 33, 33, 33, 33]

    xa = xr.DataArray(
        a,
        name="test_a",
        dims=("t",),
        attrs={"p1": 1, "nodata": 0},
        coords={"t": np.arange(a.shape[0])},
    )
    xb = xa + 0
    xb.data[:] = b
    xab = gap_fill(xa, xb)
    assert xab.name == xa.name
    assert xab.attrs == xa.attrs
    assert xab.data.tolist() == [11, 33, 33, 33, 33]

    xa.attrs["nodata"] = 11
    assert gap_fill(xa, xb).data.tolist() == [33, 0, 0, 0, 0]

    a = np.zeros((5,), dtype="float32")
    a[1:] = np.nan
    b = np.empty_like(a)
    b[:] = 33
    ab = _gap_fill_np(a, b, np.nan)

    assert ab.dtype == a.dtype
    assert ab.tolist() == [0, 33, 33, 33, 33]

    xa = xr.DataArray(
        a,
        name="test_a",
        dims=("t",),
        attrs={"p1": 1},
        coords={"t": np.arange(a.shape[0])},
    )
    xb = xa + 0
    xb.data[:] = b
    xab = gap_fill(xa, xb)
    assert xab.name == xa.name
    assert xab.attrs == xa.attrs
    assert xab.data.tolist() == [0, 33, 33, 33, 33]

    xa = xr.DataArray(
        da.from_array(a),
        name="test_a",
        dims=("t",),
        attrs={"p1": 1},
        coords={"t": np.arange(a.shape[0])},
    )

    xb = xr.DataArray(
        da.from_array(b),
        name="test_a",
        dims=("t",),
        attrs={"p1": 1},
        coords={"t": np.arange(b.shape[0])},
    )

    assert dask.is_dask_collection(xa)
    assert dask.is_dask_collection(xb)
    xab = gap_fill(xa, xb)

    assert dask.is_dask_collection(xab)
    assert xab.name == xa.name
    assert xab.attrs == xa.attrs
    assert xab.compute().values.tolist() == [0, 33, 33, 33, 33]


def test_fmask_to_bool():
    def _fake_flags(prefix="cat_", n=65):
        return {
            "bits": list(range(8)),
            "values": {str(i): f"{prefix}{i}" for i in range(0, n)},
        }

    flags_definition = {"fmask": _fake_flags()}

    fmask = xr.DataArray(
        np.arange(0, 65, dtype="uint8"), attrs={"flags_definition": flags_definition}
    )

    mm = fmask_to_bool(fmask, ("cat_1", "cat_3"))
    (ii,) = np.where(mm)
    assert tuple(ii) == (1, 3)

    # upcast to uint16 internally
    mm = fmask_to_bool(fmask, ("cat_0", "cat_15"))
    (ii,) = np.where(mm)
    assert tuple(ii) == (0, 15)

    # upcast to uint32 internally
    mm = fmask_to_bool(fmask, ("cat_1", "cat_3", "cat_31"))
    (ii,) = np.where(mm)
    assert tuple(ii) == (1, 3, 31)

    # upcast to uint64 internally
    mm = fmask_to_bool(fmask, ("cat_0", "cat_32", "cat_37", "cat_63"))
    (ii,) = np.where(mm)
    assert tuple(ii) == (0, 32, 37, 63)

    with pytest.raises(ValueError):
        fmask_to_bool(fmask, ("cat_64"))

    mm = fmask_to_bool(fmask.chunk(3), ("cat_0",)).compute()
    (ii,) = np.where(mm)
    assert tuple(ii) == (0,)

    mm = fmask_to_bool(fmask.chunk(3), ("cat_31", "cat_63")).compute()
    (ii,) = np.where(mm)
    assert tuple(ii) == (31, 63)

    # check _get_enum_values
    flags_definition = {"cat": _fake_flags("cat_"), "dog": _fake_flags("dog_")}
    assert _get_enum_values(("cat_0",), flags_definition) == (0,)
    assert _get_enum_values(("cat_0", "cat_12"), flags_definition) == (0, 12)
    assert _get_enum_values(("dog_0", "dog_13"), flags_definition) == (0, 13)
    assert _get_enum_values(("dog_0", "dog_13"), flags_definition, flag="dog") == (
        0,
        13,
    )

    with pytest.raises(ValueError) as e:
        _get_enum_values(("cat_10", "_nope"), flags_definition)
    assert "Can not find flags definitions" in str(e)

    with pytest.raises(ValueError) as e:
        _get_enum_values(("cat_10", "bah", "dog_0"), flags_definition, flag="dog")
    assert "cat_10" in str(e)


def test_enum_to_mask():
    nmax = 129

    def _fake_flags(prefix="cat_", n=nmax + 1):
        return {
            "bits": list(range(8)),
            "values": {str(i): f"{prefix}{i}" for i in range(0, n)},
        }

    flags_definition = {"fmask": _fake_flags()}

    fmask_no_flags = xr.DataArray(np.arange(0, nmax + 1, dtype="uint16"))
    fmask = xr.DataArray(
        np.arange(0, nmax + 1, dtype="uint16"),
        attrs={"flags_definition": flags_definition},
    )

    mm = enum_to_bool(fmask, ("cat_1", "cat_3", nmax, 33))
    (ii,) = np.where(mm)
    assert tuple(ii) == (1, 3, 33, nmax)

    mm = enum_to_bool(fmask, (0, 3, 17))
    (ii,) = np.where(mm)
    assert tuple(ii) == (0, 3, 17)

    mm = enum_to_bool(fmask_no_flags, (0, 3, 17))
    (ii,) = np.where(mm)
    assert tuple(ii) == (0, 3, 17)
    assert mm.dtype == "bool"

    mm = enum_to_bool(fmask_no_flags, (0, 3, 8, 17), dtype="uint8", value_true=255)
    (ii,) = np.where(mm == 255)
    assert tuple(ii) == (0, 3, 8, 17)
    assert mm.dtype == "uint8"

    mm = enum_to_bool(
        fmask_no_flags, (0, 3, 8, 17), dtype="uint8", value_true=255, invert=True
    )
    (ii,) = np.where(mm != 255)
    assert tuple(ii) == (0, 3, 8, 17)
    assert mm.dtype == "uint8"


def test_enum_to_mask_numexpr():
    elements = (1, 4, 23)
    mm = np.asarray([1, 2, 3, 4, 5, 23], dtype="uint8")

    np.testing.assert_array_equal(
        _enum_to_mask_numexpr(mm, elements), np.isin(mm, elements)
    )
    np.testing.assert_array_equal(
        _enum_to_mask_numexpr(mm, elements, invert=True),
        np.isin(mm, elements, invert=True),
    )

    bb8 = _enum_to_mask_numexpr(mm, elements, dtype="uint8", value_true=255)
    assert bb8.dtype == "uint8"

    np.testing.assert_array_equal(
        _enum_to_mask_numexpr(mm, elements, dtype="uint8", value_true=255) == 255,
        np.isin(mm, elements),
    )


def test_fuse_mean_np():
    data = np.array(
        [
            [[255, 255], [255, 50]],
            [[30, 40], [255, 80]],
            [[25, 52], [255, 98]],
        ]
    ).astype(np.uint8)

    slices = [data[i : i + 1] for i in range(data.shape[0])]
    out = _fuse_mean_np(*slices, nodata=255)
    assert (out == np.array([[28, 46], [255, 76]])).all()


def test_mask_cleanup_np():
    mask = np.ndarray(
        shape=(2, 2), dtype=bool, buffer=np.array([[True, False], [False, True]])
    )

    mask_filter_with_opening_dilation = [("opening", 1), ("dilation", 1)]
    result = mask_cleanup_np(mask, mask_filter_with_opening_dilation)
    expected_result = np.array(
        [[False, False], [False, False]],
    )
    assert (result == expected_result).all()

    mask_filter_opening = [("opening", 1), ("dilation", 0)]
    result = mask_cleanup_np(mask, mask_filter_opening)
    expected_result = np.array(
        [[False, False], [False, False]],
    )
    assert (result == expected_result).all()

    mask_filter_with_dilation = [("opening", 0), ("dilation", 1)]
    result = mask_cleanup_np(mask, mask_filter_with_dilation)
    expected_result = np.array(
        [[True, True], [True, True]],
    )
    assert (result == expected_result).all()

    mask_filter_with_closing = [("closing", 1), ("opening", 1), ("dilation", 1)]
    result = mask_cleanup_np(mask, mask_filter_with_closing)
    expected_result = np.array(
        [[True, True], [True, True]],
    )
    assert (result == expected_result).all()

    mask_filter_with_all_zero = [("closing", 0), ("opening", 0), ("dilation", 0)]
    result = mask_cleanup_np(mask, mask_filter_with_all_zero)
    expected_result = np.array(
        [[True, False], [False, True]],
    )
    assert (result == expected_result).all()

    invalid_mask_filter = [("oppening", 1), ("dilation", 1)]
    with pytest.raises(ValueError):
        mask_cleanup_np(mask, invalid_mask_filter)


def test_disk():
    # Test radius=1, 2D kernel (plus/cross pattern)
    result = _disk(1, 2)
    expected_result = np.array(
        [[False, True, False], [True, True, True], [False, True, False]],
    )
    assert (result == expected_result).all()
    assert result.ndim == expected_result.ndim
    assert result.shape == (3, 3)

    # Test radius=0 (single pixel)
    result = _disk(0, 2)
    expected_result = np.array([[True]])
    assert (result == expected_result).all()
    assert result.shape == (1, 1)

    # Test radius=2, 2D kernel (larger disk)
    result = _disk(2, 2)
    assert result.ndim == 2
    assert result.shape == (5, 5)
    # Center should always be True
    assert result[2, 2]
    # Corners should be False for disk of radius 2
    assert not result[0, 0]
    assert not result[0, 4]
    assert not result[4, 0]
    assert not result[4, 4]

    # Test 3D kernel
    result = _disk(1, 3)
    assert result.ndim == 3
    # Center should be True
    center = tuple(s // 2 for s in result.shape)
    assert result[center]

    # Test with decomposition='sequence' returns tuple format
    result = _disk(1, 2, decomposition="sequence")
    assert isinstance(result, tuple)
    assert len(result) > 0
    # Each element should be a tuple of (array, count)
    for arr, count in result:
        assert isinstance(arr, np.ndarray)
        assert isinstance(count, int)
        assert arr.ndim == 2

    # Test with decomposition='crosses' returns tuple format
    result = _disk(1, 2, decomposition="crosses")
    assert isinstance(result, tuple)
    assert len(result) > 0
    # Each element should be a tuple of (array, count)
    for arr, count in result:
        assert isinstance(arr, np.ndarray)
        assert isinstance(count, int)
        assert arr.ndim == 2

    # Test 3D with decomposition
    result = _disk(1, 3, decomposition="sequence")
    assert isinstance(result, tuple)
    for arr, count in result:
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
        assert isinstance(count, int)


def test_mask_cleanup():
    """Test mask_cleanup function with xarray.DataArray inputs."""
    # Create a simple binary mask as xarray DataArray
    mask_data = np.array(
        [[True, False, True], [False, True, False], [True, False, True]],
        dtype=bool,
    )
    mask = xr.DataArray(mask_data, dims=("y", "x"), name="test_mask")

    # Test with default mask_filters (opening=2, dilation=5)
    result = mask_cleanup(mask)
    assert isinstance(result, xr.DataArray)
    assert result.dims == mask.dims

    # Test with custom mask_filters
    result = mask_cleanup(mask, mask_filters=[("opening", 1)])
    assert isinstance(result, xr.DataArray)
    assert result.dims == mask.dims

    # Test with empty mask_filters list
    result = mask_cleanup(mask, mask_filters=[])
    assert isinstance(result, xr.DataArray)
    assert result.dims == mask.dims

    # Test with custom name parameter
    result = mask_cleanup(mask, name="custom_cleanup")
    assert isinstance(result, xr.DataArray)

    # Test with closing operation
    result = mask_cleanup(mask, mask_filters=[("closing", 1)])
    assert isinstance(result, xr.DataArray)
    assert result.dims == mask.dims

    # Test with dilation operation
    result = mask_cleanup(mask, mask_filters=[("dilation", 1)])
    assert isinstance(result, xr.DataArray)
    assert result.dims == mask.dims

    # Test with erosion operation
    result = mask_cleanup(mask, mask_filters=[("erosion", 1)])
    assert isinstance(result, xr.DataArray)
    assert result.dims == mask.dims

    # Test with multiple operations
    result = mask_cleanup(
        mask, mask_filters=[("opening", 1), ("dilation", 1), ("closing", 1)]
    )
    assert isinstance(result, xr.DataArray)
    assert result.dims == mask.dims

    # Test with dask array
    mask_dask = xr.DataArray(
        da.from_array(mask_data, chunks=(2, 2)), dims=("y", "x"), name="dask_mask"
    )
    result = mask_cleanup(mask_dask, mask_filters=[("opening", 1)])
    assert isinstance(result, xr.DataArray)
    # Compute result to verify it works with dask
    result_computed = result.compute()
    assert result_computed.dims == ("y", "x")

    # Test coordinates are preserved
    mask_with_coords = xr.DataArray(
        mask_data,
        dims=("y", "x"),
        coords={"y": np.arange(3), "x": np.arange(3)},
    )
    result = mask_cleanup(mask_with_coords, mask_filters=[("opening", 1)])
    assert result.dims == mask_with_coords.dims
    assert "y" in result.coords
    assert "x" in result.coords

    # Test with 3D data (time, y, x)
    mask_3d = np.array(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ],
        dtype=bool,
    )
    mask_3d_da = xr.DataArray(mask_3d, dims=("time", "y", "x"))
    result = mask_cleanup(mask_3d_da, mask_filters=[("opening", 1)])
    assert isinstance(result, xr.DataArray)
    assert result.ndim == 3


def test_binary_erosion():
    """Test binary_erosion function."""
    # Create a simple binary mask
    mask_data = np.array(
        [[False, False, False], [False, True, False], [False, False, False]],
        dtype=bool,
    )
    mask = xr.DataArray(mask_data, dims=("y", "x"))

    # Test erosion with default radius=1
    result = binary_erosion(mask)
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool
    assert result.dims == mask.dims
    # Erosion should shrink the bright region
    assert result.sum() <= mask.sum()

    # Test with larger radius
    result = binary_erosion(mask, radius=2)
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool

    # Test with decomposition
    result = binary_erosion(mask, radius=1, decomposition="sequence")
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool

    # Test with dask array
    mask_dask = xr.DataArray(da.from_array(mask_data, chunks=(2, 2)), dims=("y", "x"))
    result = binary_erosion(mask_dask, radius=1)
    assert isinstance(result, xr.DataArray)
    result_computed = result.compute()
    assert result_computed.dtype == bool


def test_binary_dilation():
    """Test binary_dilation function."""
    # Create a simple binary mask
    mask_data = np.array(
        [[False, False, False], [False, True, False], [False, False, False]],
        dtype=bool,
    )
    mask = xr.DataArray(mask_data, dims=("y", "x"))

    # Test dilation with default radius=1
    result = binary_dilation(mask)
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool
    assert result.dims == mask.dims
    # Dilation should expand the bright region
    assert result.sum() >= mask.sum()

    # Test with larger radius
    result = binary_dilation(mask, radius=2)
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool

    # Test with decomposition
    result = binary_dilation(mask, radius=1, decomposition="crosses")
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool

    # Test with dask array
    mask_dask = xr.DataArray(da.from_array(mask_data, chunks=(2, 2)), dims=("y", "x"))
    result = binary_dilation(mask_dask, radius=1)
    assert isinstance(result, xr.DataArray)
    result_computed = result.compute()
    assert result_computed.dtype == bool


def test_binary_opening():
    """Test binary_opening function."""
    # Create a binary mask with noise (small isolated points)
    mask_data = np.array(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )
    mask = xr.DataArray(mask_data, dims=("y", "x"))

    # Test opening with default radius=1
    result = binary_opening(mask)
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool
    assert result.dims == mask.dims
    # Opening should remove small objects (erosion then dilation)

    # Test with larger radius
    result = binary_opening(mask, radius=2)
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool

    # Test with decomposition
    result = binary_opening(mask, radius=1, decomposition="sequence")
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool

    # Test with dask array
    mask_dask = xr.DataArray(da.from_array(mask_data, chunks=(2, 2)), dims=("y", "x"))
    result = binary_opening(mask_dask, radius=1)
    assert isinstance(result, xr.DataArray)
    result_computed = result.compute()
    assert result_computed.dtype == bool

    # Test with 3D data
    mask_3d = np.array(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ],
        dtype=bool,
    )
    mask_3d_da = xr.DataArray(mask_3d, dims=("time", "y", "x"))
    result = binary_opening(mask_3d_da, radius=1)
    assert isinstance(result, xr.DataArray)
    assert result.ndim == 3
    assert result.dtype == bool


def test_binary_closing():
    """Test binary_closing function."""
    # Create a binary mask with holes (small isolated zeros in bright region)
    mask_data = np.array(
        [[True, True, True], [True, False, True], [True, True, True]],
        dtype=bool,
    )
    mask = xr.DataArray(mask_data, dims=("y", "x"))

    # Test closing with default radius=1
    result = binary_closing(mask)
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool
    assert result.dims == mask.dims
    # Closing should fill small holes (dilation then erosion)
    # The center hole should be filled
    assert result[1, 1]

    # Test with larger radius
    result = binary_closing(mask, radius=2)
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool

    # Test with decomposition
    result = binary_closing(mask, radius=1, decomposition="sequence")
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool

    # Test with dask array
    mask_dask = xr.DataArray(da.from_array(mask_data, chunks=(2, 2)), dims=("y", "x"))
    result = binary_closing(mask_dask, radius=1)
    assert isinstance(result, xr.DataArray)
    result_computed = result.compute()
    assert result_computed.dtype == bool
    # The hole should be filled
    assert result_computed.values[1, 1]

    # Test with 3D data
    mask_3d = np.array(
        [
            [[True, True], [True, False]],
            [[False, True], [True, True]],
        ],
        dtype=bool,
    )
    mask_3d_da = xr.DataArray(mask_3d, dims=("time", "y", "x"))
    result = binary_closing(mask_3d_da, radius=1)
    assert isinstance(result, xr.DataArray)
    assert result.ndim == 3
    assert result.dtype == bool


def test_binary_operations_with_coords():
    """Test binary operations preserve coordinates."""
    mask_data = np.array(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )
    mask = xr.DataArray(
        mask_data,
        dims=("y", "x"),
        coords={"y": np.arange(3), "x": np.arange(3)},
    )

    # Test all binary operations preserve coordinates
    for _, func in [
        ("erosion", binary_erosion),
        ("dilation", binary_dilation),
        ("opening", binary_opening),
        ("closing", binary_closing),
    ]:
        result = func(mask)
        assert result.dims == mask.dims
        assert "y" in result.coords
        assert "x" in result.coords
        assert result.dtype == bool


def test_binary_operations_edge_cases():
    """Test binary operations with edge case inputs."""
    # All False mask
    mask_all_false = xr.DataArray(np.zeros((3, 3), dtype=bool), dims=("y", "x"))
    result = binary_erosion(mask_all_false)
    assert result.dtype == bool
    assert not result.any()

    # All True mask
    mask_all_true = xr.DataArray(np.ones((3, 3), dtype=bool), dims=("y", "x"))
    result = binary_dilation(mask_all_true)
    assert result.dtype == bool
    assert result.all()

    # Single pixel mask
    mask_single = xr.DataArray(np.array([[True]], dtype=bool), dims=("y", "x"))
    for func in [binary_erosion, binary_dilation, binary_opening, binary_closing]:
        result = func(mask_single)
        assert isinstance(result, xr.DataArray)
        assert result.dtype == bool
