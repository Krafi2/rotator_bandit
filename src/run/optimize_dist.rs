use std::path::Path;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{agent::ThompsonSampler, experiment::ExperimentOpts};

pub struct DistParamOpts {
    pub domain: [std::ops::Range<f32>; 2],
    pub width: usize,
    pub height: usize,
    pub samples: u32,
}

/// Perform a grid search over the probability distribution parameters
pub fn dist_params(input: &Path, output: &Path, opts: DistParamOpts, eopts: ExperimentOpts) {
    // Prepare the output directory
    if !output.exists() {
        std::fs::create_dir(&output).expect("Failed to create output directory");
    }

    let seed = eopts.seed;
    let arms = eopts.shape.iter().product::<u32>() as usize;
    let experiment = super::new_experiment(input, eopts);

    let width = opts.width;
    let height = opts.height;
    let domain = opts.domain;

    let xstep = (domain[0].end - domain[0].start) / width as f32;
    let ystep = (domain[1].end - domain[1].start) / height as f32;
    let origin = [domain[0].start, domain[1].start];

    let mut buffer = vec![0.; width * height];
    let progress = std::sync::atomic::AtomicU32::default();
    let samples = width * height;

    // # Evaluate samples

    println!("Evaluating samples:");
    println!("0%");
    buffer
        .iter_mut()
        .enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|(i, val)| {
            let x = i % width;
            let y = height - i / width - 1;
            let mut experiment = experiment.clone();
            let beta = origin[1] + y as f32 * ystep;
            let alpha = origin[0] + x as f32 * xstep;
            let mut reward = 0.;
            for sample in 0..opts.samples {
                let agent = ThompsonSampler::new(arms, alpha, beta);
                experiment.reset_with_agent(agent, seed | sample as u64);
                experiment.run();
                reward += experiment.reward();
            }
            *val = reward / opts.samples as f32;

            let progress = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if 100 * progress / samples as u32 != 100 * (progress + 1) / samples as u32 {
                println!("{}%", 100 * (progress + 1) / samples as u32);
            }
        });

    let mut min = f32::MAX;
    let mut max = 0.;
    let mut max_alpha = 0.;
    let mut max_beta = 0.;
    for y in 0..height {
        let beta = origin[1] + y as f32 * ystep;
        for x in 0..width {
            let alpha = origin[0] + x as f32 * xstep;
            let val = buffer[(height - y - 1) * width + x];
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
                max_alpha = alpha;
                max_beta = beta;
            }
        }
    }

    let colorscheme = colorous::VIRIDIS;
    let scale = (max - min).recip();
    let buffer = buffer
        .into_iter()
        .flat_map(|v| {
            colorscheme
                .eval_continuous(((v - min) * scale) as f64)
                .into_array()
        })
        .collect();

    let image = image::RgbImage::from_vec(width as u32, height as u32, buffer)
        .expect("Failed to create image");

    image
        .save(output.join("dist.png"))
        .expect("Failed to write image");
    println!("Optimization finished!");
    println!("Optimum found at alpha: {max_alpha}, beta: {max_beta}");
}
