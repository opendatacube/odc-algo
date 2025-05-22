import time

import numpy as np
import xarray as xr

from odc.algo import geomedian_with_mads
from odc.algo._geomedian import geomedian

if __name__ == "__main__":
    print("Creating test data...")
    np.random.seed(42)

    # Smaller test for validation
    test_shape = (400, 400, 8, 100)  # Smaller for testing
    test_data = np.random.random(test_shape).astype(np.float32)
    mask = np.random.random(test_shape) < 0.02

    test_data[mask] = np.nan

    start_time = time.time()
    result = geomedian(test_data, num_threads=4)
    taken = time.time() - start_time
    print(f"Took {taken} seconds")

    # Test Geomedian With Mads
    xr_data = xr.DataArray(
        test_data,
        dims=("y", "x", "band", "time"),
        coords={
            "y": np.arange(400),
            "x": np.arange(400),
            "band": np.arange(8),
            "time": np.arange(100),
        },
    )

    gm, mads = geomedian_with_mads(xr_data)
