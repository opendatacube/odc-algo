use ndarray::{Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Ix1, Ix2};
use num_traits::identities::Zero;
use std::cmp::{PartialOrd, PartialEq};


pub fn percentile<T: PartialOrd + PartialEq + Zero + Clone + Copy + ValidCheck>(
    in_array: ArrayView<T, Ix2>,
    percentiles: ArrayView<f64, Ix1>,
    nodata: T,
) -> Array<T, Ix2> {

    let cols = in_array.shape()[1];
    let num_perc = percentiles.shape()[0];
    let mut out_array: Array<T, Ix2> = ArrayBase::zeros([num_perc, cols]);

    percentile_chunk(in_array, out_array.view_mut(), percentiles, nodata);
    
    out_array
}


fn percentile_chunk<T: PartialOrd + PartialEq + Zero + Clone + Copy + ValidCheck>(
    in_array: ArrayView<T, Ix2>,
    mut out_array: ArrayViewMut<T, Ix2>,
    percentiles: ArrayView<f64, Ix1>,
    nodata: T,
) {

    let mut data: Vec<T> = Vec::with_capacity(in_array.shape()[0]);

    let iter = in_array.axis_iter(Axis(1)).zip(out_array.axis_iter_mut(Axis(1)));
    
    for (in_array, mut out_array) in iter {
        
        data.clear();

        for x in in_array.iter() {
            if x.is_valid(nodata) {
                data.push(*x);
            }
        }

        if data.len() > 3 {
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for (y, p) in out_array.iter_mut().zip(percentiles.iter()) {

                let idx = (*p * ((data.len() - 1) as f64)).round() as usize;
                *y = data[idx];
            }
        } else {
            for y in out_array.iter_mut() {
                *y = nodata;
            }
        }
    }

}


pub trait ValidCheck {
    fn is_valid(&self, nodata: Self) -> bool;
}


impl ValidCheck for i16 {
    fn is_valid(&self, nodata: i16) -> bool {
        *self != nodata
    }
}


impl ValidCheck for u16 {
    fn is_valid(&self, nodata: u16) -> bool {
        *self != nodata
    }
}


impl ValidCheck for i8 {
    fn is_valid(&self, nodata: i8) -> bool {
        *self != nodata
    }
}


impl ValidCheck for u8 {
    fn is_valid(&self, nodata: u8) -> bool {
        *self != nodata
    }
}


impl ValidCheck for f32 {
    fn is_valid(&self, _nodata: f32) -> bool {
        !self.is_nan()
    }
}


impl ValidCheck for f64 {
    fn is_valid(&self, _nodata: f64) -> bool {
        !self.is_nan()
    }
}
