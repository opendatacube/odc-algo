# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2025 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native load and masking."""

from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    cast,
)

import xarray as xr
from datacube import Datacube
from datacube.testutils.io import native_geobox
from odc.geo.geobox import GeoBox, pad as gbox_pad
from ._grouper import group_by_nothing, solar_offset
from ._masking import _max_fuser, _nodata_fuser, _or_fuser, enum_to_bool, mask_cleanup
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from datacube.model import Dataset


def compute_native_load_geobox(
    dst_geobox: GeoBox, ds: Dataset, band: str, buffer: float | None = None
) -> GeoBox:
    """Compute area of interest for a given Dataset given query.

    Take native projection and resolution from ``ds, band`` pair and compute
    region in that projection that fully encloses footprint of the
    ``dst_geobox`` with some padding. Construct GeoBox that encloses that
    region fully with resolution/pixel alignment copied from supplied band.

    :param dst_geobox:
    :param ds: Sample dataset (only resolution and projection is used, not footprint)
    :param band: Reference band to use
                 (resolution of output GeoBox will match resolution of this band)
    :param buffer: Buffer in units of CRS of ``ds`` (meters usually),
                   default is 10 pixels worth
    """
    native: GeoBox = native_geobox(ds, basis=band)
    if buffer is None:
        buffer = 10 * cast(float, max(map(abs, (native.resolution.y, native.resolution.x))))  # type: ignore

    assert native.crs is not None
    return GeoBox.from_geopolygon(
        dst_geobox.extent.to_crs(native.crs).buffer(buffer),
        crs=native.crs,
        resolution=native.resolution,
        align=native.alignment,
    )


def choose_transform_path(
    src_crs: str,
    dst_crs: str,
    transform_code: str | None = None,
    area_of_interest: Sequence[float] | None = None,
) -> str:
    # leave gdal to choose the best option if nothing is specified
    if transform_code is None and area_of_interest is None:
        return {}

    if area_of_interest is not None:
        assert len(area_of_interest) == 4
        area_of_interest = aoi.AreaOfInterest(*area_of_interest)

    transformer_group = transformer.TransformerGroup(
        src_crs, dst_crs, area_of_interest=area_of_interest
    )
    if transform_code is None:
        return {"COORDINATE_OPERATION": transformer_group.transformers[0].to_proj4()}
    for t in transformer_group.transformers:
        for step in json.loads(t.to_json()).get("steps", []):
            if step.get("type", "") == "Transformation":
                authority_code = step.get("id", {})
                if transform_code.split(":")[0].upper() in authority_code.get(
                    "authority", ""
                ) and transform_code.split(":")[1] == str(
                    authority_code.get("code", "")
                ):
                    return {"COORDINATE_OPERATION": t.to_proj4()}
    # raise error if nothing is available
    raise ValueError(f"Not able to find transform path by {transform_code}")


def _split_by_grid(xx: xr.DataArray) -> list[xr.DataArray]:
    def extract(grid_id, ii):
        yy = xx[ii]
        crs = xx.grid2crs[grid_id]
        yy.attrs.update(crs=crs)
        yy.attrs.pop("grid2crs", None)
        return yy

    return [extract(grid_id, ii) for grid_id, ii in xx.groupby(xx.grid).groups.items()]


def _native_load_1(
    sources: xr.DataArray,
    bands: tuple[str, ...],
    geobox: GeoBox,
    optional_bands: Optional[Tuple[str, ...]] = None,
    basis: Optional[str] = None,
    load_chunks: Optional[Dict[str, int]] = None,
    pad: Optional[int] = None,
) -> xr.Dataset:
    if basis is None:
        basis = bands[0]
    (ds,) = sources.data[0]
    load_geobox = compute_native_load_geobox(geobox, ds, basis)
    if pad is not None:
        load_geobox = gbox_pad(load_geobox, pad)

    mm = ds.product.lookup_measurements(bands)
    if optional_bands is not None:
        for ob in optional_bands:
            try:
                om = ds.product.lookup_measurements(ob)
            except KeyError:
                continue
            else:
                mm.update(om)
    xx = Datacube.load_data(sources, load_geobox, mm, dask_chunks=load_chunks)
    return xx


def native_load(
    dss: Sequence[Dataset],
    bands: Sequence[str],
    geobox: GeoBox,
    basis: Optional[str] = None,
    load_chunks: Optional[Dict[str, int]] = None,
    pad: Optional[int] = None,
):
    sources = group_by_nothing(list(dss), solar_offset(geobox.extent))
    for srcs in _split_by_grid(sources):
        _xx = _native_load_1(
            srcs,
            tuple(bands),
            geobox,
            basis=basis,
            load_chunks=load_chunks,
            pad=pad,
        )
        yield _xx


# pylint: disable=too-many-arguments, too-many-locals
def _apply_native_transform_1(
    xx: xr.Dataset,
    geobox: GeoBox,
    native_transform: Callable[[xr.Dataset], xr.Dataset],
    groupby: Optional[str] = None,
    fuser: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
) -> xr.Dataset:
    xx = native_transform(xx)

    if groupby is not None:
        if fuser is None:
            fuser = _nodata_fuser  # type: ignore
        xx = xx.groupby(groupby).map(fuser)

    return xx


def load_with_native_transform(
    dss: Sequence[Dataset],
    bands: Sequence[str],
    geobox: GeoBox,
    native_transform: Callable[[xr.Dataset], xr.Dataset],
    optional_bands: tuple[str, ...] | None = None,
    basis: str | None = None,
    groupby: str | None = None,
    fuser: Callable[[xr.Dataset], xr.Dataset] | None = None,
    resampling: str = "nearest",
    chunks: dict[str, int] | None = None,
    load_chunks: dict[str, int] | None = None,
    pad: int | None = None,
    **kw,
) -> xr.Dataset:
    """Load a bunch of datasets with native pixel transform.

    :param dss: A list of datasets to load
    :param bands: Which measurements to load
    :param geobox: GeoBox of the final output
    :param native_transform: ``xr.Dataset -> xr.Dataset`` transform,
                             should support Dask inputs/outputs
    :param basis: Name of the band to use as a reference for what is "native projection"
    :param groupby: One of 'solar_day'|'time'|'idx'|None
    :param fuser: Optional ``xr.Dataset -> xr.Dataset`` transform
    :param resampling: Any resampling mode supported by GDAL as a string:
                       nearest, bilinear, average, mode, cubic, etc...
    :param chunks: If set use Dask, must be in dictionary form
                   ``{'x': 4000, 'y': 4000}``

    :param load_chunks: Defaults to ``chunks`` but can be different if supplied
                        (different chunking for native read vs reproject)

    :param pad: Optional padding in native pixels, if set will load extra
                pixels beyond of what is needed to reproject to final
                destination. This is useful when you plan to apply convolution
                filter or morphological operators on input data.

    :param kw: Used to support old names ``dask_chunks`` and ``group_by``
               also kwargs for reproject ``tranform_code`` in the form of
               "authority:code", e.g., "epsg:9688", and ``area_of_interest``,
               e.g., [-180, -90, 180, 90]

    1. Partition datasets by native Projection
    2. For every group do
       - Load data
       - Apply native_transform
       - [Optional] fuse rasters that happened on the same day/time
       - Reproject to final geobox
    3. Stack output of (2)
    4. [Optional] fuse rasters that happened on the same day/time
    """
    if fuser is None:
        fuser = _nodata_fuser

    if groupby is None:
        groupby = kw.pop("group_by", "idx")

    if chunks is None:
        chunks = kw.pop("dask_chunks", None)

    if load_chunks is None:
        load_chunks = chunks

    _xx = []
    # fail if the intended transform not available
    # to avoid any unexpected results
    for xx in native_load(dss, bands, geobox, basis, load_chunks, pad):
        extra_args = choose_transform_path(
            xx.crs,
            geobox.crs,
            kw.pop("transform_code"),
            kw.pop("area_of_interest"),
        )
        extra_args.update(kw)

        yy = _apply_native_transform_1(
            _apply_native_transform_1(
                xx,
                geobox,
                native_transform,
                optional_bands=tuple(optional_bands),
                basis=basis,
                resampling=resampling,
                groupby=groupby,
                fuser=fuser,
            )

        _xx += [
        xr_reproject(
            yy, geobox, resampling=resampling, chunks=chunks, **extra_args,
        )  # type: ignore
        ]

    if len(_xx) == 1:
        xx = _xx[0]
    else:
        xx = xr.concat(_xx, _xx[0].dims[0])  # type: ignore
        if groupby != "idx":
            xx = xx.groupby(groupby).map(fuser)
    # TODO: probably want to replace spec MultiIndex with just `time` component
    return xx


def load_enum_mask(
    dss: list[Dataset],
    band: str,
    geobox: GeoBox,
    categories: Iterable[str | int],
    invert: bool = False,
    resampling: str = "nearest",
    groupby: str | None = None,
    chunks: dict[str, int] | None = None,
    **kw,
) -> xr.DataArray:
    """Load enumerated mask (like fmask).

    1. Load each mask time slice separately in native projection of the file
    2. Convert enum to Boolean (F:0, T:255)
    3. Optionally (groupby='solar_day') group observations on the same day
       using OR for pixel fusing: T,F->T
    4. Reproject to destination GeoBox (any resampling mode is ok)
    5. Optionally group observations on the same day using OR for pixel fusing T,F->T
    6. Finally convert to real Bool
    """

    def native_op(ds):
        return ds.map(
            enum_to_bool,
            categories=categories,
            invert=invert,
            dtype="uint8",
            value_true=255,
        )

    xx = load_with_native_transform(
        dss,
        (band,),
        geobox,
        native_op,
        basis=band,
        resampling=resampling,
        groupby=groupby,
        chunks=chunks,
        fuser=_max_fuser,
        **kw,
    )
    return xx[band] > 127


def load_enum_filtered(
    dss: Sequence[Dataset],
    band: str,
    geobox: GeoBox,
    categories: Iterable[str | int],
    filters: Iterable[tuple[str, int]] | None = None,
    groupby: str | None = None,
    resampling: str = "nearest",
    chunks: dict[str, int] | None = None,
    **kw,
) -> xr.DataArray:
    """Load enumerated mask (like fmask/SCL) with native pixel filtering.

    The idea is to load "cloud" classes while adding some padding, then erase
    pixels that were classified as cloud in any of the observations on a given
    day.

    This method converts enum-mask to a boolean image in the native projection
    of the data and then reprojects boolean image to the final
    projections/resolution. This allows one to use any resampling strategy,
    like ``average`` or ``cubic`` and not be limited to a few resampling
    strategies that support operations on categorical data.

    :param dss: A list of datasets to load
    :param band: Which measurement band to load
    :param geobox: GeoBox of the final output
    :param categories: Enum values or names

    :param filters: iterable tuples of morphological operations in the order
                    you want them to perform, e.g., [("opening", 2), ("dilation", 5)]
    :param groupby: One of 'solar_day'|'time'|'idx'|None
    :param resampling: Any resampling mode supported by GDAL as a string:
                       nearest, bilinear, average, mode, cubic, etc...
    :param chunks: If set use Dask, must be in dictionary form
                   ``{'x': 4000, 'y': 4000}``
    :param kw: Passed on to ``load_with_native_transform``


    1. Load each mask time slice separately in native projection of the file
    2. Convert enum to Boolean
    3. Optionally (groupby='solar_day') group observations on the same day
       using OR for pixel fusing: T,F->T
    4. Optionally apply ``mask_cleanup`` in native projection (after fusing)
    4. Reproject to destination GeoBox (any resampling mode is ok)
    5. Optionally group observations on the same day using OR for pixel fusing T,F->T
    """

    def native_op(xx: xr.Dataset) -> xr.Dataset:
        _xx = enum_to_bool(xx[band], categories)
        return xr.Dataset(
            {band: _xx},
            attrs={"native": True},  # <- native flag needed for fuser
        )

    def fuser(xx: xr.Dataset) -> xr.Dataset:
        """Fuse with OR.

        Fuse with OR, and when fusing in native pixel domain apply mask_cleanup if
        requested
        """
        is_native = xx.attrs.get("native", False)
        xx = xx.map(_or_fuser)
        xx.attrs.pop("native", None)

        if is_native and filters is not None:
            _xx = xx[band]
            assert isinstance(_xx, xr.DataArray)
            xx[band] = mask_cleanup(_xx, mask_filters=filters)

        return xx

    # unless set by user to some value use largest filter radius for pad value
    pad: int | None = kw.pop("pad", None)
    if pad is None:
        if filters is not None:
            pad = max(list(zip(*filters, strict=False))[1])  # type: ignore

    xx = load_with_native_transform(
        dss,
        (band,),
        geobox,
        native_op,
        fuser=fuser,
        groupby=groupby,
        resampling=resampling,
        chunks=chunks,
        pad=pad,
        **kw,
    )[band]
    assert isinstance(xx, xr.DataArray)
    return xx
