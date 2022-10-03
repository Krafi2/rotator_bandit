use std::{cell::Cell, path::Path, rc::Rc};

use crate::{
    agent::Agent,
    experiment::{CellId, Experiment, ExperimentOpts, Video},
};

/// Perform an experiment
pub fn experiment(
    input: &Path,
    output: &Path,
    experiments: u32,
    graph_points: u32,
    opts: ExperimentOpts,
) {
    // # Prepare output

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

    // # Load data

    let rot_len = opts.shape.iter().product::<u32>();
    let logger = Rc::new(Cell::new(Some(crate::agent::Logger::new(
        &output.join("regret.csv"),
        experiments * opts.trials * rot_len / graph_points,
    ))));

    // Load the video data and wrap the agents in a logger
    let videos = super::load_data(input.as_ref())
        .map(|(path, thumbs)| {
            let agent = crate::agent::ThompsonSampler::new(thumbs.len(), opts.alpha, opts.beta);
            Video::new(
                thumbs,
                crate::agent::RegretLogger::new(agent, logger.clone()),
                path,
            )
        })
        .collect();

    let clickability = super::clickability(&opts.shape);
    let mut experiment = Experiment::new(videos, clickability, opts);

    // # Run experiments

    for e in 0..experiments {
        experiment.run();
        let img = rotator_snapshot(&mut experiment);
        img.save(output.join(format!("thumbnails-{:02}.png", e)))
            .expect("Failed to write thumbnails");
        println!("Finished experiment {}", e);
    }

    // # Print output

    let mut logger = logger.take().expect("Logfile is in use");

    println!("Total reward: {}", logger.reward());
    logger.flush().expect("Failed to flush log file");

    let click_estimate = (0..rot_len)
        .map(|cell| {
            experiment
                .click_estimate()
                .weight(CellId(cell as usize))
                .recip()
        })
        .collect::<Vec<_>>();

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

    // Spawn a python process to generate the plot
    let exit = std::process::Command::new("python3")
        .args([
            "scripts/regret.py".into(),
            output.join("regret.csv"),
            output.join("regret.png"),
        ])
        .spawn()
        .and_then(|mut child| child.wait())
        .expect("Failed to generate regret plot");

    // Check if the script succeeded
    if exit.code().is_none() || exit.code().unwrap() != 0 {
        println!("Regret graph generation failed!");
    }

    println!("Done!");
}

/// Get an image of the rotator
fn rotator_snapshot<A: Agent>(experiment: &mut Experiment<A>) -> image::RgbImage {
    // Get a snapshot of the rotator
    let impressions = experiment.generate_impressions();

    let sample_thumb = impressions[0].0;
    // Open a sample thumbnail to find the thumbnail image size
    let sample_thumb =
        image::open(experiment.videos()[sample_thumb.video()].thumb_path(sample_thumb))
            .expect("Failed to read image");
    let (img_width, img_height) = image::GenericImageView::dimensions(&sample_thumb);
    let [width, height] = experiment.shape().clone();

    let mut image = image::RgbImage::new(width * img_width, height * img_height);

    // Tile the thumbnail
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
