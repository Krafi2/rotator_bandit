mod agent;
mod experiment;
mod optimizer;

use crate::experiment::{Clickability, Experiment, Thumbnail, Video};
use agent::{Agent, RegretLogger, ThompsonSampler};
use image::GenericImageView;
use std::io::Write;
use std::{
    cell::Cell,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, BufWriter},
    path::{Path, PathBuf},
    rc::Rc,
};

pub type DefaultRng = rand_chacha::ChaCha8Rng;

/// A seed to initialize random behaviour
const SEED: u64 = 0;
const SHAPE: [u32; 2] = [5, 10];
const TRIALS: u32 = 10_00;
const EXPERIMENTS: u32 = 1;
const GRAPH_POINTS: u32 = 600;
const TEST_PROB: f64 = 1.;

const INITIAL_SAMPLES: u32 = 64;
const SAMPLES: u32 = 2;
const RETAIN: f64 = 0.5;
const REFINE: u32 = 0;
const DOMAIN: [std::ops::Range<f64>; 2] = [0f64..1f64, 0f64..1f64];

enum Op {
    Optimize,
    Experiment,
}

fn main() {
    let mut args = std::env::args();
    let _ = args.next();

    let op = args
        .next()
        .map(|op| match op.as_str() {
            "optimize" => Op::Optimize,
            other => panic!("Unrecognized operation '{}'", other),
        })
        .unwrap_or(Op::Experiment);

    match op {
        Op::Optimize => optimize(args),
        Op::Experiment => experiment(args),
    }
}

fn optimize(mut args: std::env::Args) {
    // The input data folder
    let input = args
        .next()
        .expect("Expected an input argument")
        .parse::<PathBuf>()
        .unwrap();

    // The output data folder
    let output = args
        .next()
        .expect("Expected an output argument")
        .parse::<PathBuf>()
        .unwrap();

    // Prepare the output directory
    if !output.exists() {
        std::fs::create_dir(&output).expect("Failed to create output directory");
    }

    let videos = load_data(input.as_ref())
        .map(|(path, thumbs)| {
            let agent = ThompsonSampler::new(thumbs.len(), 1., 1.);
            Video::new(thumbs, agent, path)
        })
        .collect();

    let clickability = clickability(SHAPE);
    let experiment = Experiment::new(videos, SHAPE, TRIALS, clickability, TEST_PROB, SEED);

    let point_path = output.join("optimizer.csv");
    let mut point_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .open(&point_path)
            .expect("Failed to open optimizer point cache"),
    );

    // Compute the total number of samples that will be collected with the given parameters
    let n = (INITIAL_SAMPLES.pow(2) as f64 * (SAMPLES.pow(2) as f64 * RETAIN).powi(REFINE as i32))
        as u32;
    let step = (n / 100).max(1);
    let mut i = 0;
    let mut max = 0.;
    let mut max_point = vec![0.; 2];

    println!("Collecting {n} samples:");

    optimizer::grid_search(&DOMAIN, INITIAL_SAMPLES, SAMPLES, RETAIN, REFINE, |point| {
        if i % step == 0 {
            println!("{}%", i / step);
        }
        i += 1;

        let alpha = point[0].powi(4) * 100. + 1e-6;
        let beta = point[1] * 200. + 1e-6;

        let mut experiment = experiment.clone();
        for vid in experiment.videos_mut() {
            let arms = vid.thumbs().len();
            vid.replace_agent(ThompsonSampler::new(arms, alpha as f32, beta as f32));
        }
        experiment.run();
        let val = experiment.reward() as f64;

        if val > max {
            max = val;
            max_point.clone_from_slice(&point);
        }

        writeln!(&mut point_writer, "{},{},{val}", point[0], point[1]).unwrap();

        val
    });
    point_writer.flush().unwrap();

    println!("Optimization finished!");
    println!(
        "Optimum found at alpha: {}, beta: {}",
        max_point[0], max_point[1]
    );

    println!("Generating optimizer plot...");
    let exit = std::process::Command::new("python3")
        .args([
            "scripts/optimizer.py".into(),
            output.join("optimizer.csv"),
            output.join("optimizer.png"),
        ])
        .spawn()
        .and_then(|mut child| child.wait())
        .expect("Failed to generate regret plot");

    if exit.code().is_none() || exit.code().unwrap() != 0 {
        println!("Optimizer plot generation failed!");
    }
    println!("Done!");
}

fn experiment(mut args: std::env::Args) {
    // The input data folder
    let input = args
        .next()
        .expect("Expected an input argument")
        .parse::<PathBuf>()
        .unwrap();

    // The output data folder
    let output = args
        .next()
        .expect("Expected an output argument")
        .parse::<PathBuf>()
        .unwrap();

    // Prepare the output directory
    if !output.exists() {
        std::fs::create_dir(&output).expect("Failed to create output directory");
    }

    // Only clean the output folder if the path is relative to avoid accidentaly deleting important
    // files.
    if output.is_relative() {
        for entry in std::fs::read_dir(&output).expect("Failed to open output directory") {
            if let Ok(entry) = entry {
                // Delete only the thumbnail files
                if entry.file_name().to_string_lossy().contains("thumbnails-") {
                    if let Err(err) = std::fs::remove_file(entry.path()) {
                        eprintln!("Error removing file: {}", err);
                    }
                }
            }
        }
    }

    let logger = Rc::new(Cell::new(Some(agent::Logger::new(
        &output.join("regret.csv"),
        EXPERIMENTS * TRIALS * SHAPE.iter().product::<u32>() / GRAPH_POINTS,
    ))));

    let videos = load_data(input.as_ref())
        .map(|(path, thumbs)| {
            let agent = ThompsonSampler::new(thumbs.len(), 1., 1.);
            Video::new(thumbs, RegretLogger::new(agent, logger.clone()), path)
        })
        .collect();

    let clickability = clickability(SHAPE);
    let mut experiment = Experiment::new(videos, SHAPE, TRIALS, clickability, TEST_PROB, SEED);

    for e in 0..EXPERIMENTS {
        experiment.run();
        let img = rotator_snapshot(&mut experiment);
        img.save(output.join(format!("thumbnails-{:02}.png", e)))
            .expect("Failed to write thumbnails");
        println!("Finished experiment {}", e);
    }
    let mut logger = logger.take().expect("Logfile is in use");

    println!("Total reward: {}", logger.reward());
    logger.flush().expect("Failed to flush log file");

    // Normalize clickability estimates
    let max = experiment
        .click_estimate()
        .iter()
        .map(Clickability::ratio)
        .reduce(f32::max)
        .unwrap()
        .max(f32::EPSILON);
    let click_estimate = experiment
        .click_estimate()
        .iter()
        .map(|click| click.ratio() / max);

    println!("Clickability matrix estimate:");
    println!(" N |   real  | estimate");
    for (i, (real, estimate)) in experiment
        .clickability()
        .into_iter()
        .zip(click_estimate)
        .enumerate()
    {
        println!("{:02} | {:.5} | {:.5}", i, real, estimate);
    }

    println!("Generating regret plot...");

    let exit = std::process::Command::new("python3")
        .args([
            "scripts/regret.py".into(),
            output.join("regret.csv"),
            output.join("regret.png"),
        ])
        .spawn()
        .and_then(|mut child| child.wait())
        .expect("Failed to generate regret plot");

    if exit.code().is_none() || exit.code().unwrap() != 0 {
        println!("Regret graph generation failed!");
    }

    println!("Done!");
}

fn load_data(input: &Path) -> impl Iterator<Item = (PathBuf, Vec<Thumbnail>)> {
    std::fs::read_dir(input)
        .expect("Failed to read dir")
        .map(|video| {
            let path = video.unwrap().path();
            let ctrs = path.join("ctrs.txt");

            // Parse the thumbnail data from the csv file
            let thumbs = BufReader::new(File::open(ctrs).expect("Failed to read input"))
                .lines()
                .map(|s| {
                    let s = s.expect("Failed to read line");
                    let sub = s.split(',').collect::<Vec<_>>();

                    Thumbnail::new(sub[0].parse().unwrap(), sub[1].parse().unwrap())
                })
                .collect::<Vec<_>>();
            (path, thumbs)
        })
}

fn clickability(shape: [u32; 2]) -> Vec<f32> {
    let mat_len = shape.iter().product::<u32>() as usize;
    (0..mat_len)
        .map(|i| f32::powf(0.1, i as f32 / (mat_len - 1) as f32))
        .collect()
}

fn rotator_snapshot<A: Agent>(experiment: &mut Experiment<A>) -> image::RgbImage {
    let (impressions, _) = experiment.generate_impressions();

    let sample_thumb = impressions[0].0;
    let sample_thumb =
        image::open(experiment.videos()[sample_thumb.video()].thumb_path(sample_thumb))
            .expect("Failed to read image");
    let (img_width, img_height) = sample_thumb.dimensions();
    let [width, height] = SHAPE;

    let mut image = image::RgbImage::new(width * img_width, height * img_height);

    for y in 0..height {
        for x in 0..width {
            let n = y * width + x;
            let id = impressions[n as usize].0;
            let path = experiment.videos()[id.video()].thumb_path(id);
            let thumb = image::open(path).expect("Failed to read image").into_rgb8();
            image::imageops::overlay(
                &mut image,
                &thumb,
                (x * img_width) as i64,
                (y * img_height) as i64,
            );
        }
    }

    image
}
