use crate::{agent::ThompsonSampler, experiment::ExperimentOpts};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

pub struct TestpOpts {
    pub res: usize,
    pub domain: std::ops::Range<f32>,
    pub samples: u32,
}

/// Perform a grid search over the test probability
pub fn testp(input: &Path, output: &Path, opts: TestpOpts, eopts: ExperimentOpts) {
    // Prepare the output directory
    if !output.exists() {
        std::fs::create_dir(&output).expect("Failed to create output directory");
    }

    let seed = eopts.seed;
    let arms = eopts.shape.iter().product::<u32>() as usize;
    let alpha = eopts.alpha;
    let beta = eopts.beta;
    let experiment = super::new_experiment(input, eopts);

    let res = opts.res;
    let domain = opts.domain;
    let step = (domain.end - domain.start) / res as f32;

    let progress = std::sync::atomic::AtomicU32::default();

    // # Evaluate samples

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

    // # Print results

    let mut writer = BufWriter::new(
        File::options()
            .create(true)
            .write(true)
            .truncate(true)
            .open(output.join("testp.csv"))
            .expect("Failed to open file"),
    );

    let mut max = 0.;
    let mut max_p = 0.;
    for (i, &r) in rewards.iter().enumerate() {
        let testp = domain.start + i as f32 * step;
        writeln!(&mut writer, "{},{}", testp, r).unwrap();
        if r > max {
            max = r;
            max_p = testp;
        }
    }

    println!("Optimum found at testp: {max_p}");
    println!("Generating  plot...");

    writer.flush().unwrap();

    // Spawn a python process to generate the plot
    let exit = std::process::Command::new("python3")
        .args([
            "scripts/plot_testp.py".into(),
            output.join("testp.csv"),
            output.join("testp.png"),
        ])
        .spawn()
        .and_then(|mut child| child.wait())
        .expect("Failed to generate plot");

    // Check if the script succeeded
    if exit.code().is_none() || exit.code().unwrap() != 0 {
        println!("Failed to generate plot!");
    }

    println!("Done!");
}
