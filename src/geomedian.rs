//! Array based Geometric Median implementations supporting Missing Values
use ndarray::parallel::prelude::*;
use ndarray::{s, Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Ix1, Ix2, Ix3, Ix4, Zip};
use num_traits::identities::Zero;

use crate::mad;
use rayon;

/// Geometric Median on a 4-d array of floats
pub fn geomedian<'a>(
    in_array: ArrayView<'a, f32, Ix4>,
    maxiters: usize,
    eps: f32,
    num_threads: usize,
    scale: f32, 
    offset: f32,
) -> (Array<f32, Ix3>, Array<f32, Ix3>) {
    let rows = in_array.shape()[0];
    let columns = in_array.shape()[1];
    let bands = in_array.shape()[2];

    let mut gm: Array<f32, Ix3> = ArrayBase::zeros([rows, columns, bands]);
    let mut mads_array: Array<f32, Ix3> = ArrayBase::zeros([rows, columns, 3]);

    let iter = Zip::from(gm.axis_iter_mut(Axis(0)))
        .and(in_array.axis_iter(Axis(0)))
        .and(mads_array.axis_iter_mut(Axis(0)));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    pool.install(|| {
        iter.into_par_iter()
            .for_each(|(gm, in_arr, mads_arr)| {
                geomedian_column(in_arr, gm, mads_arr, maxiters, eps, scale, offset)
            })
    });

    (gm, mads_array)
}

fn geomedian_column<'a>(
    in_array: ArrayView<'a, f32, Ix3>,
    mut gm: ArrayViewMut<f32, Ix2>,
    mut mads_array: ArrayViewMut<f32, Ix2>,
    maxiters: usize,
    eps: f32,
    scale: f32,
    offset: f32,
) {
    let shape = in_array.shape();
    let columns = shape[0];
    let bands = shape[1];
    let time_steps = shape[2];

    let mut data: Array<f32, Ix2> = ArrayBase::zeros([time_steps, bands]);

    for column in 0..columns {
        let in_array = in_array.index_axis(Axis(0), column);
        let mut gm = gm.index_axis_mut(Axis(0), column);
        let mut mads_array = mads_array.index_axis_mut(Axis(0), column);

        let mut data: ArrayViewMut<f32, Ix2> = get_valid_data(in_array, &mut data, scale, offset);

        if data.shape()[0] == 0 {
            gm.fill(f32::NAN);
            mads_array.fill(f32::NAN);
            continue;
        }

        // seed initialization
        gm.assign(&data.mean_axis(Axis(0)).unwrap());
        geomedian_pixel(data.view(), &mut gm, maxiters, eps);

        let inv_scale = 1.0 / scale;
        
        for band in 0..bands {
            gm[[band]] = inv_scale * (gm[[band]] - offset); 
        }

        for t in 0..data.shape()[0] {
            for band in 0..bands {
                data[[t, band]] = inv_scale * (data[[t, band]] - offset);
            }
        }

        // make immutable view of `gm` (implementds `Copy` trait)
        let gm = gm.view();
        let data = data.view();
        mads_array[[0]] = mad::emad(data, gm);
        mads_array[[1]] = mad::smad(data, gm);
        mads_array[[2]] = mad::bcmad(data, gm);
    }
}

fn get_valid_data<'a>(
    in_array: ArrayView<f32, Ix2>,
    data: &'a mut Array<f32, Ix2>,
    scale: f32,
    offset: f32,
) -> ArrayViewMut<'a, f32, Ix2> {
    // copies the valid data for each (row, column) data from `in_array` to `valid_data
    // and tranposes the band and time dimensions for better cache perfomance

    let bands = in_array.shape()[0];
    let time_steps = in_array.shape()[1];
    let mut idx: usize = 0;
    
    for t in 0..time_steps {
        let mut valid = true;
        for band in 0..bands {
            valid &= !in_array[[band, t]].is_nan();
            if !valid {
                break;
            }
            data[[idx, band]] = scale * in_array[[band, t]] + offset;
        }

        idx += valid as usize;
        
    }

    data.slice_mut(s![0..idx, ..])
}

/// Geometric Median on a 4-d array of Ints
pub fn geomedian_int<'a, T: NumericTools + std::cmp::PartialEq + Clone + Copy + Zero + Sync + Send>(
    in_array: ArrayView<'a, T, Ix4>,
    maxiters: usize,
    eps: f32,
    num_threads: usize,
    nodata: T,
    scale: f32,
    offset: f32,
) -> (Array<T, Ix3>, Array<f32, Ix3>) {
    let rows = in_array.shape()[0];
    let columns = in_array.shape()[1];
    let bands = in_array.shape()[2];

    let mut gm: Array<T, Ix3> = ArrayBase::zeros([rows, columns, bands]);
    let mut mads_array: Array<f32, Ix3> = ArrayBase::zeros([rows, columns, 3]);

    let iter = Zip::from(gm.axis_iter_mut(Axis(0)))
        .and(in_array.axis_iter(Axis(0)))
        .and(mads_array.axis_iter_mut(Axis(0)));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    pool.install(|| {
        iter.into_par_iter().for_each(|(gm, in_arr, mads_arr)| {
            geomedian_column_int(in_arr, gm, mads_arr, maxiters, eps, nodata, scale, offset)
        })
    });

    (gm, mads_array)
}

fn geomedian_column_int<'a, T: NumericTools + std::cmp::PartialEq + Clone + Copy>(
    in_array: ArrayView<'a, T, Ix3>,
    mut gm: ArrayViewMut<T, Ix2>,
    mut mads_array: ArrayViewMut<f32, Ix2>,
    maxiters: usize,
    eps: f32,
    nodata: T,
    scale: f32,
    offset: f32,
) {
    let shape = in_array.shape();
    let columns = shape[0];
    let bands = shape[1];
    let time_steps = shape[2];

    let mut gm_f32: Array<f32, Ix1> = ArrayBase::zeros([bands]);
    let mut data: Array<f32, Ix2> = ArrayBase::zeros([time_steps, bands]);

    for column in 0..columns {
        let in_array = in_array.index_axis(Axis(0), column);
        let mut gm = gm.index_axis_mut(Axis(0), column);
        let mut mads_array = mads_array.index_axis_mut(Axis(0), column);

        let mut data: ArrayViewMut<f32, Ix2> = get_valid_data_int(in_array, &mut data, nodata, scale, offset);

        if data.shape()[0] == 0 {
            gm.fill(nodata);
            mads_array.fill(f32::NAN);
            continue;
        }

        // seed initialization
        gm_f32.assign(&data.mean_axis(Axis(0)).unwrap());
        geomedian_pixel(data.view(), &mut gm_f32.view_mut(), maxiters, eps);
        
        let inv_scale = 1.0 / scale;
        for band in 0..bands {
            let val = (gm_f32[[band]] - offset) * inv_scale;
            gm_f32[[band]] = val; // should this value be rounded?
            gm[[band]] = T::from_f32(val.round());
        }

        for t in 0..data.shape()[0] {
            for band in 0..bands {
                data[[t, band]] = (data[[t, band]] - offset) * inv_scale;
            }
        }

        // make immutable view of `gm` (implementds `Copy` trait)
        let gm_f32 = gm_f32.view();
        let data = data.view();
        mads_array[[0]] = mad::emad(data, gm_f32);
        mads_array[[1]] = mad::smad(data, gm_f32);
        mads_array[[2]] = mad::bcmad(data, gm_f32);

    }
}

fn get_valid_data_int<'a, T: NumericTools + Copy + std::cmp::PartialEq>(
    in_array: ArrayView<T, Ix2>,
    data: &'a mut Array<f32, Ix2>,
    nodata: T,
    scale: f32, 
    offset: f32,
) -> ArrayViewMut<'a, f32, Ix2> {
    // copies the valid data for each (row, column) data from `in_array` to `valid_data
    // and tranposes the band and time dimensions for better cache perfomance

    let bands = in_array.shape()[0];
    let time_steps = in_array.shape()[1];
    let mut idx: usize = 0;

    for t in 0..time_steps {
        let mut valid = true;
        for band in 0..bands {
            valid &= in_array[[band, t]] != nodata;
            if !valid {
                break;
            }
            data[[idx, band]] = scale * NumericTools::to_f32(in_array[[band, t]]) + offset;
        }

        idx += valid as usize;
    }

    data.slice_mut(s![0..idx, ..])
}


fn geomedian_pixel(
    data: ArrayView<f32, Ix2>,
    gm: &mut ArrayViewMut<f32, Ix1>,
    maxiters: usize,
    eps: f32,
) {
    let time_steps = data.shape()[0];
    let bands = data.shape()[1];
    let mut temp_median: Vec<f32> = vec![0.0; bands];

    // main loop
    for _ in 0..maxiters {
        temp_median.iter_mut().for_each(|x| *x = 0.0);
        let mut inv_dist_sum: f32 = 0.0;

        for t in 0..time_steps {
            let mut dist: f32 = 0.0;
            for band in 0..bands {
                dist += (gm[[band]] - data[[t, band]]).powi(2);
            }

            dist = dist.sqrt();

            let mut inv_dist: f32 = 0.0;
            if dist > 0.0 {
                inv_dist = 1.0 / dist;
            }

            inv_dist_sum += inv_dist;

            for band in 0..bands {
                temp_median[band] += data[[t, band]] * inv_dist;
            }
        }

        // check improvement between iterations
        // exit if smaller than tolerance
        let mut change: f32 = 0.0;
        for band in 0..bands {
            temp_median[band] /= inv_dist_sum;
            change += (&temp_median[band] - gm[[band]]).powi(2);
            gm[[band]] = temp_median[band];
        }

        if change.sqrt() < eps {
            break;
        }
    }
}

pub trait NumericTools {
    fn to_f32(n: Self) -> f32;
    fn from_f32(n: f32) -> Self;
}

impl NumericTools for i16 {
    fn to_f32(n: i16) -> f32 {
        n as f32
    }
    fn from_f32(n: f32) -> i16 {
        n.round() as i16
    }

}

impl NumericTools for u16 {
    fn to_f32(n: u16) -> f32 {
        n as f32
    }
    fn from_f32(n: f32) -> u16 {
        n.round() as u16
    }
}
