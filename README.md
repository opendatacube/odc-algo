## Open Data Cube Algorithms

[Xarray] and [Dask] friendly EO Processing Algorithms and Utilities.

> **Note:** This package only contains algorithms. If you want to use them for processing
> EO data, you'll find using [odc-stats] much simpler.


- Cloud Masking
- Geometric Median
- Percentiles
- Dask Aware Raster Reprojection
- Efficiently generating and saving Cloud Optimized GeoTIFFs to S3
- Reshaping Dask Arrays for Efficient Computation
- Converting between Floats with NaNs and Ints with nodata values


[Dask]: https://www.dask.org/
[Xarray]: https://docs.xarray.dev/en/stable/
[odc-stats]: https://github.com/opendatacube/odc-stats

## Installation

```
pip install odc-algo
```

## Usage

## Building

1. Install the Python build tool. `python -m pip install build`
2. Build this package. `python -m build`


## Development

1. Follow build instructions
2. Install as dev `pip install -e .[dev]`

Alternatively, install with the whl file.


# Tasks

- [ ] Decide whether to use [pixi], or uv. I think pixi, it handles Rust stuff.
- [ ] Document the Geomedian API we're trying to expose. See [odc-stats]
- [ ] Document the Percentile API we're exposing.
- [ ] Regresssion Tests instead of installing old dependencies like hdstats.
- [ ] Update GitHub Actions to build and test Rust backend
- [ ] Update GitHub Actions to build and test against multiple Python versions
- [ ] Consider what type of binary wheels to build. [abi3/multi-python version compatible is tempting]
- [ ] Update Rust dependencies.
- [ ] De-duplicate with [odc-geo]. It includes COG and Warp functionality that's better maintained, but may not be identical...
- [ ] Consider vendoring the skimage morphology functions we're using, instead of depending on the whole thing.

[odc-geo]: https://github.com/opendatacube/odc-geo
[pixi]: https://pixi.sh/latest/
