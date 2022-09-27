use crate::{
    agent::ThompsonSampler,
    experiment::{Experiment, ExperimentOpts, Video},
};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
};

fn new_experiment(input: &Path, eopts: ExperimentOpts) -> Experiment<ThompsonSampler> {
    let videos = crate::experiment::load_data(input.as_ref())
        .map(|(path, thumbs)| {
            let agent = ThompsonSampler::new(thumbs.len(), 1., 1.);
            Video::new(thumbs, agent, path)
        })
        .collect();

    let clickability = crate::experiment::clickability(&eopts.shape);
    Experiment::new(videos, clickability, eopts)
}

pub struct DistParamOpts {
    pub domain: [std::ops::Range<f32>; 2],
    pub width: usize,
    pub height: usize,
    pub samples: u32,
}

pub fn dist_params(input: &Path, output: &Path, opts: DistParamOpts, eopts: ExperimentOpts) {
    // Prepare the output directory
    if !output.exists() {
        std::fs::create_dir(&output).expect("Failed to create output directory");
    }

    let seed = eopts.seed;
    let arms = eopts.shape.iter().product::<u32>() as usize;
    let experiment = new_experiment(input, eopts);

    let width = opts.width;
    let height = opts.height;
    let domain = opts.domain;

    let xstep = (domain[0].end - domain[0].start) / width as f32;
    let ystep = (domain[1].end - domain[1].start) / height as f32;
    let origin = [domain[0].start, domain[1].start];

    let mut buffer = vec![0.; width * height];
    let progress = std::sync::atomic::AtomicU32::default();

    println!("Evaluating samples:");
    println!("0%");
    buffer
        .chunks_mut(width)
        .rev()
        .enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|(y, row)| {
            let mut experiment = experiment.clone();
            let beta = origin[1] + y as f32 * ystep;
            for (x, val) in row.into_iter().enumerate() {
                let alpha = origin[0] + x as f32 * xstep;
                let mut reward = 0.;
                for sample in 0..opts.samples {
                    let agent = ThompsonSampler::new(arms, alpha, beta);
                    experiment.reset_with_agent(agent, seed | sample as u64);
                    experiment.run();
                    reward += experiment.reward();
                }
                *val = reward / opts.samples as f32;
            }
            let progress = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if 100 * progress / height as u32 != 100 * (progress + 1) / height as u32 {
                println!("{}%", 100 * (progress + 1) / height as u32);
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
            let val = buffer[y * width + x];
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
        .save(output.join("optimizer.png"))
        .expect("Failed to write image");
    println!("Optimization finished!");
    println!("Optimum found at alpha: {max_alpha}, beta: {max_beta}");
}

pub struct TestpOpts {
    pub res: usize,
    pub domain: std::ops::Range<f32>,
    pub samples: u32,
}

pub fn testp(input: &Path, output: &Path, opts: TestpOpts, eopts: ExperimentOpts) {
    // Prepare the output directory
    if !output.exists() {
        std::fs::create_dir(&output).expect("Failed to create output directory");
    }

    let seed = eopts.seed;
    let arms = eopts.shape.iter().product::<u32>() as usize;
    let alpha = eopts.alpha;
    let beta = eopts.beta;
    let experiment = new_experiment(input, eopts);

    let res = opts.res;
    let domain = opts.domain;
    let step = (domain.end - domain.start) / res as f32;

    let progress = std::sync::atomic::AtomicU32::default();

    println!("Evaluating samples:");
    println!("0%");
    let rewards = (0..res)
        .into_par_iter()
        .map(|i| {
            let mut experiment = experiment.clone();
            let testp = domain.start + i as f32 * step;
            experiment.set_testp(testp as f64);
            let mut reward = 0.;
            for sample in 0..opts.samples {
                let agent = ThompsonSampler::new(arms, alpha, beta);
                experiment.reset_with_agent(agent, seed | sample as u64);
                experiment.run();
                reward += experiment.reward();
            }
            let progress = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if 100 * progress / res as u32 != 100 * (progress + 1) / res as u32 {
                println!("{}%", 100 * (progress + 1) / res as u32);
            }
            reward / opts.samples as f32
        })
        .collect::<Vec<_>>();

    let mut writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .open(output.join("testp.csv"))
            .expect("Failed to open file"),
    );

    let mut max = 0.;
    let mut max_p = 0.;
    for (i, &r) in rewards.iter().enumerate() {
        writeln!(&mut writer, "{}", r).unwrap();
        if r > max {
            max = r;
            max_p = domain.start + i as f32 * step;
        }
    }

    println!("Optimum found at testp: {max_p}");
    println!("Generating  plot...");

    let exit = std::process::Command::new("python3")
        .args([
            "scripts/plot_testp.py".into(),
            output.join("testp.csv"),
            output.join("testp.png"),
        ])
        .spawn()
        .and_then(|mut child| child.wait())
        .expect("Failed to generate plot");

    if exit.code().is_none() || exit.code().unwrap() != 0 {
        println!("Failed to generate plot!");
    }

    println!("Done!");
}
