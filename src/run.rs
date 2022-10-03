mod experiment;
mod optimize_dist;
mod optimize_testp;

pub use experiment::experiment;
pub use optimize_dist::{dist_params, DistParamOpts};
pub use optimize_testp::{testp, TestpOpts};

use std::{
    io::BufRead,
    path::{Path, PathBuf},
};

use crate::{
    agent::ThompsonSampler,
    experiment::{Experiment, ExperimentOpts, Thumb, Video},
};

/// Load the dataset
fn load_data(input: &Path) -> impl Iterator<Item = (PathBuf, Vec<Thumb>)> {
    std::fs::read_dir(input)
        .expect("Failed to read dir")
        .map(|video| {
            let path = video.unwrap().path();
            let ctrs = path.join("ctrs.txt");

            // Parse the thumbnail data from the csv file
            let thumbs =
                std::io::BufReader::new(std::fs::File::open(ctrs).expect("Failed to read input"))
                    .lines()
                    .map(|s| {
                        let s = s.expect("Failed to read line");
                        let sub = s.split(',').collect::<Vec<_>>();

                        Thumb::new(sub[0].parse().unwrap(), sub[1].parse().unwrap())
                    })
                    .collect::<Vec<_>>();
            (path, thumbs)
        })
}

/// Generate the "ground truth" clickability
fn clickability(shape: &[u32; 2]) -> Vec<f32> {
    let mat_len = shape.iter().product::<u32>() as usize;
    (0..mat_len).map(|i| f32::powf(0.99, i as f32)).collect()
}

/// Construct a new experiment with no logging capabilities
fn new_experiment(input: &Path, eopts: ExperimentOpts) -> Experiment<ThompsonSampler> {
    let videos = load_data(input.as_ref())
        .map(|(path, thumbs)| {
            let agent = ThompsonSampler::new(thumbs.len(), 1., 1.);
            Video::new(thumbs, agent, path)
        })
        .collect();

    let clickability = clickability(&eopts.shape);
    Experiment::new(videos, clickability, eopts)
}
