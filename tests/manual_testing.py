import datacube
from datacube.utils.dask import start_local_dask

from odc.algo import geomedian_with_mads
import os

def main():
    dc = datacube.Datacube()

    data = dc.load(
        product='ga_ls8c_ard_3',
        time='2015',
        x=(1497580, 1502260),
        y=(-3958185 , -3954961),
        crs='EPSG:3577',
        measurements=[
            "nbart_red",
            "nbart_green",
            "nbart_blue",
            "nbart_nir",
            "nbart_swir_1",
            "nbart_swir_2",
        ],
        dask_chunks=dict(x=100, y=100),
        group_by="solar_day",
    )
    from dask.distributed import performance_report, Client

    client = Client(processes=False)
    with performance_report(filename="/home/omad/stats_outputs/dask-report.html"):
        ## some dask computation
        result = geomedian_with_mads(data)

        result.to_netcdf("/home/omad/stats_outputs/manual_test.nc")

def more_real_test():
    from odc.stats.plugins.gm import StatsGMLS
    from datacube.utils.geometry import GeoBox, box

    dc = datacube.Datacube()
    datasets = dc.find_datasets(product='ga_ls8c_ard_3',
                                time=('2015-01-01', '2015-04-01'),
                                x=(1497580, 1502260),
                                y=(-3958185, -3954961),
                                crs='EPSG:3577',
                                measurements=[
                                    "nbart_red",
                                    "nbart_green",
                                    "nbart_blue",
                                    "nbart_nir",
                                    "nbart_swir_1",
                                    "nbart_swir_2",
                                ],
                                )

    geobox = GeoBox.from_geopolygon(geopolygon=box(left=1497580, right=1502260, top=-3958185, bottom=-3954961, crs="EPSG:3577"), resolution=[30,30])

    proc = StatsGMLS()

    from dask.distributed import performance_report, Client

    client = start_local_dask(
        threads_per_worker=4, processes=False, memory_limit="20Gi"
    )
    with performance_report(filename="/home/omad/stats_outputs/dask-report.html"):
        ds = proc.reduce(
            proc.input_data(
                datasets,
                geobox,
                transform_code=proc.transform_code,
                area_of_interest=proc.area_of_interest,
            )
        )
        print(ds)
        ds = ds.compute()
        print(ds)
        ds.to_netcdf("/home/omad/stats_outputs/manual_test_2.nc")

if __name__ == '__main__':
    os.environ['DATACUBE_DB_URL'] = 'postgresql://sandbox_reader:f0GelTszC2Umj1uu@localhost:5444/odc'
    more_real_test()
#    main()
