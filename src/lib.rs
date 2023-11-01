//! Fast Array Summaries for Use from Python for Raster Earth Observation Data
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, ToPyArray};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::prelude::*;

mod geomedian;
mod mad;
mod percentile;

#[pyfunction]
#[pyo3(name = "_geomedian")]
fn py_geomedian<'a>(
    py: Python<'a>,
    in_array: &'a PyArray4<f32>,
    maxiters: usize,
    eps: f32,
    num_threads: usize,
    scale: f32,
    offset: f32,
) -> (&'a PyArray3<f32>, &'a PyArray3<f32>) {
    let in_array = in_array.readonly();
    let in_array = in_array.as_array();

    // release GIL for call to geomedian
    let (gm, mads) = py
        .allow_threads(|| geomedian::geomedian(in_array, maxiters, eps, num_threads, scale, offset));

    (gm.to_pyarray(py), mads.to_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "_geomedian_int16")]
fn py_geomedian_int16<'a>(
    py: Python<'a>,
    in_array: &'a PyArray4<i16>,
    maxiters: usize,
    eps: f32,
    num_threads: usize,
    nodata: i16,
    scale: f32,
    offset: f32,
) -> (&'a PyArray3<i16>, &'a PyArray3<f32>) {
    let in_array = in_array.readonly();
    let in_array = in_array.as_array();

    // release GIL for call to geomedian
    let (gm, mads) =
        py.allow_threads(|| geomedian::geomedian_int(in_array, maxiters, eps, num_threads, nodata, scale, offset));

    (gm.to_pyarray(py), mads.to_pyarray(py))
}
#[pyfunction]
#[pyo3(name="_geomedian_uint16")]
fn py_geomedian_uint16<'a>(
    py: Python<'a>,
    in_array: &'a PyArray4<u16>,
    maxiters: usize,
    eps: f32,
    num_threads: usize,
    nodata: u16,
    scale: f32,
    offset: f32,
) -> (&'a PyArray3<u16>, &'a PyArray3<f32>) {
    let in_array = in_array.readonly();
    let in_array = in_array.as_array();

    // release GIL for call to geomedian
    let (gm, mads) =
        py.allow_threads(|| geomedian::geomedian_int(in_array, maxiters, eps, num_threads, nodata, scale, offset));

    (gm.to_pyarray(py), mads.to_pyarray(py))
}

#[pyfunction]
#[pyo3(name="_percentile_uint16")]
fn py_percentile_uint16<'a>(
    py: Python<'a>,
    in_array: &'a PyArray2<u16>,
    percentiles: &'a PyArray1<f64>,
    nodata: u16,
) -> &'a PyArray2<u16> {
    let in_array = in_array.readonly();
    let in_array = in_array.as_array();
    
    let percentiles = percentiles.readonly();
    let percentiles = percentiles.as_array();

    // release GIL
    let out = py.allow_threads( || 
        percentile::percentile(in_array, percentiles, nodata)
    );

    out.to_pyarray(py)
}


#[pyfunction]
#[pyo3(name="_percentile_int16")]
fn py_percentile_int16<'a>(
    py: Python<'a>,
    in_array: &'a PyArray2<i16>,
    percentiles: &'a PyArray1<f64>,
    nodata: i16,
) -> &'a PyArray2<i16> {
    let in_array = in_array.readonly();
    let in_array = in_array.as_array();
    
    let percentiles = percentiles.readonly();
    let percentiles = percentiles.as_array();

    // release GIL
    let out = py.allow_threads( || 
        percentile::percentile(in_array, percentiles, nodata)
    );

    out.to_pyarray(py)
}


#[pyfunction]
#[pyo3(name="_percentile_uint8")]
fn py_percentile_uint8<'a>(
    py: Python<'a>,
    in_array: &'a PyArray2<u8>,
    percentiles: &'a PyArray1<f64>,
    nodata: u8,
) -> &'a PyArray2<u8> {
    let in_array = in_array.readonly();
    let in_array = in_array.as_array();
    
    let percentiles = percentiles.readonly();
    let percentiles = percentiles.as_array();

    // release GIL
    let out = py.allow_threads( || 
        percentile::percentile(in_array, percentiles, nodata)
    );

    out.to_pyarray(py)
}


#[pyfunction]
#[pyo3(name="_percentile_int8")]
fn py_percentile_int8<'a>(
    py: Python<'a>,
    in_array: &'a PyArray2<i8>,
    percentiles: &'a PyArray1<f64>,
    nodata: i8,
) -> &'a PyArray2<i8> {
    let in_array = in_array.readonly();
    let in_array = in_array.as_array();
    
    let percentiles = percentiles.readonly();
    let percentiles = percentiles.as_array();

    // release GIL
    let out = py.allow_threads( || 
        percentile::percentile(in_array, percentiles, nodata)
    );

    out.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name="_percentile_f32")]
fn py_percentile_f32<'a>(
    py: Python<'a>,
    in_array: &'a PyArray2<f32>,
    percentiles: &'a PyArray1<f64>,
) -> &'a PyArray2<f32> {
    let in_array = in_array.readonly();
    let in_array = in_array.as_array();
    
    let percentiles = percentiles.readonly();
    let percentiles = percentiles.as_array();

    // release GIL
    let out = py.allow_threads( || 
        percentile::percentile(in_array, percentiles, f32::NAN)
    );

    out.to_pyarray(py)
}


#[pyfunction]
#[pyo3(name="_percentile_f64")]
fn py_percentile_f64<'a>(
    py: Python<'a>,
    in_array: &'a PyArray2<f64>,
    percentiles: &'a PyArray1<f64>,
) -> &'a PyArray2<f64> {
    let in_array = in_array.readonly();
    let in_array = in_array.as_array();
    
    let percentiles = percentiles.readonly();
    let percentiles = percentiles.as_array();

    // release GIL
    let out = py.allow_threads( || 
        percentile::percentile(in_array, percentiles, f64::NAN)
    );

    out.to_pyarray(py)
}

#[pymodule]
fn backend(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(py_geomedian, m)?)?;
    m.add_function(wrap_pyfunction!(py_geomedian_int16, m)?)?;
    m.add_function(wrap_pyfunction!(py_geomedian_uint16, m)?)?;

    m.add_function(wrap_pyfunction!(py_percentile_int8, m)?)?;
    m.add_function(wrap_pyfunction!(py_percentile_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(py_percentile_int16, m)?)?;
    m.add_function(wrap_pyfunction!(py_percentile_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(py_percentile_f32, m)?)?;
    m.add_function(wrap_pyfunction!(py_percentile_f64, m)?)?;


    Ok(())
}
