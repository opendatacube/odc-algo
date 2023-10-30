

import pytest
import xarray
from odc.algo import geomedian_with_mads

scale = 1 / 10_000
chunks = (-1, 100, 100)
cfg = dict(
    maxiters=1000,
    num_threads=1,
    scale=scale,
    offset=-1 * scale,
    reshape_strategy="mem",
    out_chunks=(-1, -1, -1),
#    work_chunks=chunks,
    compute_count=True,
    compute_mads=True,
)
def test_geomedian_with_mads(xr_regression, datadir):
    ds = xarray.open_dataset(datadir / "landsat8_2020_ard.nc", chunks="auto")

    result = geomedian_with_mads(ds, **cfg)
    result = result.compute()
    xr_regression(result)


@pytest.fixture
def xr_regression(request, datadir, original_datadir):
    """
    Inspired by pytest-regressions, but for xarray datasets
    """
    name = request.node.name

    def check(test_data):
        filename = f"{name}.nc"
        if (original_datadir / filename).exists():
            data = xarray.open_dataset(datadir / filename)
            xarray.testing.assert_allclose(data, test_data)
        else:
            test_data.to_netcdf(original_datadir / filename, 
                                encoding={
                                    vname: {'complevel': 9, 'compression':'zstd'} 
                                    for vname in list(test_data)})

    return check
