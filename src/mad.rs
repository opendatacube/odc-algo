use ndarray::{ArrayView, Ix1, Ix2};

pub fn emad(data: ArrayView<f32, Ix2>, gm: ArrayView<f32, Ix1>) -> f32 {
    let time_steps = data.shape()[0];
    let bands = data.shape()[1];
    let mut dists: Vec<f32> = vec![0.0; time_steps];

    for t in 0..time_steps {
        for band in 0..bands {
            dists[t] += (data[[t, band]] - gm[[band]]).powi(2);
        }
        dists[t] = dists[t].sqrt();
    }

    median(dists)
}

pub fn smad(data: ArrayView<f32, Ix2>, gm: ArrayView<f32, Ix1>) -> f32 {
    let time_steps = data.shape()[0];
    let bands = data.shape()[1];
    let mut dists: Vec<f32> = vec![0.0; time_steps];

    let mut norm_1 = 0.0;
    for band in 0..bands {
        norm_1 += gm[[band]].powi(2);
    }
    norm_1 = norm_1.sqrt();

    for t in 0..time_steps {
        let mut norm_2 = 0.0;

        for band in 0..bands {
            norm_2 += data[[t, band]].powi(2);
            dists[t] += data[[t, band]] * gm[[band]];
        }
        norm_2 = norm_2.sqrt();
        dists[t] = 1.0 - (dists[t] / (norm_1 * norm_2));
    }

    median(dists)
}

pub fn bcmad(data: ArrayView<f32, Ix2>, gm: ArrayView<f32, Ix1>) -> f32 {
    let time_steps = data.shape()[0];
    let bands = data.shape()[1];
    let mut dists: Vec<f32> = vec![0.0; time_steps];

    for t in 0..time_steps {
        let mut denom = 0.0;

        for band in 0..bands {
            dists[t] += (data[[t, band]] - gm[[band]]).abs();
            denom += (data[[t, band]] + gm[[band]]).abs();
        }
        dists[t] /= denom;
    }

    median(dists)
}

fn median(mut data: Vec<f32>) -> f32 {
    // Filter out NaN values before sorting
    data.retain(|&x| !x.is_nan());

    // return NaN if no valid values
    if data.len() == 0 {
        return std::f32::NAN;
    }

    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = data.len() / 2;

    if data.len() % 2 == 0 {
        return 0.5 * (data[idx] + data[idx - 1]);
    } else {
        return data[idx];
    }
}
