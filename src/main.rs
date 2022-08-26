mod agent;
mod experiment;

use crate::experiment::{Clickability, Experiment, Thumbnail, Video};
use agent::{RegretLogger, ThompsonSampler};
use image::GenericImageView;
use std::{
    cell::Cell,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    rc::Rc,
};

/// A seed to initialize random behaviour
const SEED: u64 = 0;
const SHAPE: [u32; 2] = [5, 10];
const TRIALS: u32 = 10000000;
const EXPERIMENTS: u32 = 1;
const GRAPH_POINTS: u32 = 600;
const TEST_PROB: f64 = 1.;

fn main() {
    // The input data folder
    let input = std::env::args().nth(1).expect("Expected an input argument");
    // The output data folder
    let output = std::env::args()
        .nth(2)
        .expect("Expected an output argument")
        .parse::<PathBuf>()
        .unwrap();

    // Only clean the output folder if the path is relative to avoid accidentaly deleting important
    // files.
    if output.exists() && output.is_relative() {
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

    let videos = std::fs::read_dir(input)
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

            let agent = ThompsonSampler::new(thumbs.len());

            Video::new(thumbs, RegretLogger::new(agent, logger.clone()), path)
        })
        .collect::<Vec<_>>();

    let mat_len = SHAPE.iter().product::<u32>() as usize;
    let clickability = (1..=mat_len)
        .map(|i| 1. - (i as f32 / mat_len as f32).sqrt())
        .collect();
    let mut experiment = Experiment::new(videos, SHAPE, TRIALS, clickability, TEST_PROB, SEED);

    for e in 0..EXPERIMENTS {
        experiment.run();
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
            .save(output.join(format!("thumbnails-{:02}.png", e)))
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
    println!("N | real | estimate");
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
