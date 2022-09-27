use crate::agent::BetaParams;

use super::agent::{Action, Agent, Reward};
use super::DefaultRng;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng, SeedableRng,
};
use rand_distr::Distribution;
use std::cell::Cell;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// The data of a thumbnail
#[derive(Debug, Clone)]
pub struct Thumbnail {
    /// The id of the thumbnail
    pub id: u32,
    /// Click through ratio
    pub ctr: f64,
    /// The number of clicks that the thumbnail has received
    pub clicks: u32,
    /// The number of impressions that the thumbnail has received
    pub impressions: u32,
}

impl Thumbnail {
    pub fn new(id: u32, ctr: f64) -> Self {
        Self {
            id,
            ctr,
            clicks: 0,
            impressions: 0,
        }
    }

    /// Register an imression
    fn impress(&mut self, click: bool) {
        self.impressions += 1;
        self.clicks += click as u32;
    }

    fn reset(&mut self) {
        self.clicks = 0;
        self.impressions = 0;
    }
}

pub struct Video<A> {
    thumbs: Vec<Thumbnail>,
    agent: A,
    path: PathBuf,
}

impl<A: Clone> Clone for Video<A> {
    fn clone(&self) -> Self {
        Self {
            thumbs: self.thumbs.clone(),
            agent: self.agent.clone(),
            path: self.path.clone(),
        }
    }
}

impl<A: std::fmt::Debug> std::fmt::Debug for Video<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Video")
            .field("thumbs", &self.thumbs)
            .field("agent", &self.agent)
            .field("path", &self.path)
            .finish()
    }
}

impl<A> Video<A> {
    pub fn new(thumbs: Vec<Thumbnail>, agent: A, path: PathBuf) -> Self {
        Self {
            thumbs,
            agent,
            path,
        }
    }

    pub fn thumb_path(&self, id: ThumbId) -> PathBuf {
        self.path.join(format!("{:02}.png", self.thumbs[id.1].id))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct VideoId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThumbId(VideoId, usize);

impl ThumbId {
    pub fn video(&self) -> usize {
        self.0 .0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellId(usize);

#[derive(Debug, Clone)]
pub struct Clickability {
    clicks: Vec<u32>,
    impressions: Vec<u32>,
    scale: f32,
}

impl Clickability {
    fn new(n: usize) -> Self {
        Self {
            clicks: vec![1; n],
            impressions: vec![1; n],
            scale: 1.,
        }
    }

    fn update(&mut self, cell: CellId, click: bool) {
        let id = cell.0;
        self.clicks[id] += click as u32;
        self.impressions[id] += 1;
        if id == 0 {
            self.scale = self.clicks[id] as f32 / self.impressions[id] as f32;
        }
    }

    pub fn weight(&self, cell: CellId) -> f32 {
        let id = cell.0;
        self.impressions[id] as f32 / self.clicks[id] as f32 * self.scale
    }

    fn reset(&mut self) {
        for clicks in &mut self.clicks {
            *clicks = 1;
        }
        for imp in &mut self.impressions {
            *imp = 1;
        }
        self.scale = 1.;
    }
}

#[derive(Debug, Clone)]
struct Ratings {
    alpha: f32,
    beta: f32,
    dist_params: Vec<BetaParams>,
    dist: Vec<rand_distr::Beta<f32>>,
}

impl Ratings {
    fn new(n: usize, alpha: f32, beta: f32) -> Self {
        let params = BetaParams { alpha, beta };
        Self {
            alpha,
            beta,
            dist_params: vec![params.clone(); n],
            dist: vec![params.new_dist(); n],
        }
    }

    fn reset(&mut self) {
        *self = Self::new(self.dist.len(), self.alpha, self.beta)
    }

    fn update(&mut self, id: VideoId, reward: Reward) {
        let dist = &mut self.dist_params[id.0];
        dist.update(reward);
        self.dist[id.0] = dist.new_dist();
    }

    fn choose(&self, n: usize, rng: &mut DefaultRng) -> Vec<VideoId> {
        let mut ratings = self
            .dist
            .iter()
            .enumerate()
            .map(|(id, dist)| (VideoId(id), dist.sample(rng)))
            .collect::<Vec<_>>();
        ratings.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).expect("Unexpected NaN"));
        ratings[0..n].into_iter().map(|(id, _)| *id).collect()
    }
}

pub struct ExperimentOpts {
    pub seed: u64,
    pub shape: [u32; 2],
    pub trials: u32,
    pub test_prob: f64,
    pub alpha: f32,
    pub beta: f32,
}

pub struct Experiment<A> {
    ratings: Ratings,
    // Video data
    videos: Vec<Video<A>>,
    // Shape of the rotator
    shape: [u32; 2],
    // Number of trials per experiment
    trials: u32,
    // The real clickability matrix
    clickability: Vec<f32>,
    // Estimate of the clickability matrix
    click_estimate: Clickability,
    // Probabity of a test trial
    test_prob: f64,
    // The accumulated reward
    reward: f32,
    rng: DefaultRng,
}

impl<A: std::fmt::Debug> std::fmt::Debug for Experiment<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Experiment")
            .field("videos", &self.videos)
            .field("shape", &self.shape)
            .field("trials", &self.trials)
            .field("clickability", &self.clickability)
            .field("click_estimate", &self.click_estimate)
            .field("test_prob", &self.test_prob)
            .field("reward", &self.reward)
            .field("rng", &self.rng)
            .finish()
    }
}

impl<A: Clone> Clone for Experiment<A> {
    fn clone(&self) -> Self {
        Self {
            ratings: self.ratings.clone(),
            videos: self.videos.clone(),
            shape: self.shape.clone(),
            trials: self.trials.clone(),
            clickability: self.clickability.clone(),
            click_estimate: self.click_estimate.clone(),
            test_prob: self.test_prob.clone(),
            reward: self.reward.clone(),
            rng: self.rng.clone(),
        }
    }
}

impl<A: Agent> Experiment<A> {
    pub fn new(videos: Vec<Video<A>>, clickability: Vec<f32>, opts: ExperimentOpts) -> Self {
        Self {
            ratings: Ratings::new(videos.len(), opts.alpha, opts.beta),
            videos,
            clickability,
            click_estimate: Clickability::new(opts.shape.iter().product::<u32>() as usize),
            shape: opts.shape,
            trials: opts.trials,
            test_prob: opts.test_prob,
            reward: 0.,
            rng: DefaultRng::seed_from_u64(opts.seed),
        }
    }

    fn update_thumb(&mut self, thumb: ThumbId, cell: CellId, click: bool, test: bool) {
        // Update thumbnail data
        let vid = &mut self.videos[thumb.0 .0];
        vid.thumbs[thumb.1].impress(click);

        // Update clickability
        if test {
            self.click_estimate.update(cell, click);
        }

        self.reward += click as u32 as f32;

        // Compute the normalized reward
        let reward = Reward(click, self.click_estimate.weight(cell));

        // Update the video rating
        self.ratings.update(thumb.0, reward);

        // Update agent
        vid.agent.update(Action(thumb.1), reward);
    }

    fn update(
        &mut self,
        mut impressions: Vec<(ThumbId, CellId)>,
        mut clicks: Vec<(ThumbId, CellId)>,
        test: bool,
    ) {
        impressions.sort_unstable_by_key(|(ThumbId(vid, _), _)| *vid);
        clicks.sort_unstable_by_key(|(ThumbId(vid, _), _)| *vid);
        let mut c = 0;
        let mut i = 0;

        // Separate impressions which lead to clicks from these that did not
        loop {
            let (&imp, &click) = match (impressions.get(i), clicks.get(c)) {
                // We received a click that didn't have an impression
                (None, Some(_)) => panic!("Click not contained in impressions"),
                // We ran out of clicks, so lets speed through the remaining imressions
                (Some(_), None) => {
                    for &(thumb, cell) in &impressions[i..] {
                        self.update_thumb(thumb, cell, false, test);
                    }
                    break;
                }
                // Got a click and an impression, proceed to check what to do
                (Some(imp), Some(click)) => (imp, click),
                // All done
                (None, None) => break,
            };

            match imp.0 .0.cmp(&click.0 .0) {
                // Not a click
                std::cmp::Ordering::Less => {
                    self.update_thumb(imp.0, imp.1, false, test);
                    i += 1;
                }
                // Click
                std::cmp::Ordering::Equal => {
                    self.update_thumb(click.0, click.1, true, test);
                    i += 1;
                    c += 1;
                }
                // Probably a bug
                std::cmp::Ordering::Greater => {
                    panic!("Click not contained in impressions")
                }
            }
        }
    }

    pub fn generate_test_impressions(&mut self) -> Vec<(ThumbId, CellId)> {
        let n = self.shape.iter().product::<u32>() as usize;
        // Pick the videos that will be displayed
        let mut videos = self
            .videos
            .iter_mut()
            .enumerate()
            .choose_multiple(&mut self.rng, n);
        videos.shuffle(&mut self.rng);

        videos
            .into_iter()
            .enumerate()
            .map(|(cell, (id, vid))| {
                let thumb = vid.agent.choose(&mut self.rng).0;
                (ThumbId(VideoId(id), thumb), CellId(cell))
            })
            .collect()
    }

    pub fn generate_impressions(&mut self) -> Vec<(ThumbId, CellId)> {
        let n = self.shape.iter().product::<u32>() as usize;
        self.ratings
            .choose(n, &mut self.rng)
            .into_iter()
            .enumerate()
            .map(|(cell, id)| {
                let vid = &self.videos[id.0];
                let thumb = vid.agent.choose(&mut self.rng).0;
                (ThumbId(id, thumb), CellId(cell))
            })
            .collect()
    }

    fn run_trial(&mut self) {
        let test = self.rng.gen_bool(self.test_prob);
        let impressions = if test {
            self.generate_test_impressions()
        } else {
            self.generate_impressions()
        };

        let clicks = impressions
            .iter()
            .copied()
            .filter(|(thumb, cell)| {
                self.rng.gen_bool(
                    self.videos[thumb.0 .0].thumbs[thumb.1].ctr * self.clickability[cell.0] as f64,
                )
            })
            .collect();

        self.update(impressions, clicks, test)
    }

    pub fn run(&mut self) {
        for _ in 0..self.trials {
            self.run_trial()
        }
    }

    pub fn videos(&self) -> &[Video<A>] {
        self.videos.as_ref()
    }

    pub fn clickability(&self) -> &[f32] {
        self.clickability.as_ref()
    }

    pub fn click_estimate(&self) -> &Clickability {
        &self.click_estimate
    }

    pub fn reward(&self) -> f32 {
        self.reward
    }

    pub fn set_testp(&mut self, prob: f64) {
        self.test_prob = prob;
    }
}

impl<A: Clone> Experiment<A> {
    pub fn reset_with_agent(&mut self, agent: A, seed: u64) {
        self.reward = 0.;
        self.click_estimate.reset();
        for vid in &mut self.videos {
            for thumb in &mut vid.thumbs {
                thumb.reset();
            }
            vid.agent = agent.clone();
        }
        self.ratings.reset();
        self.rng = DefaultRng::seed_from_u64(seed);
    }
}

pub fn experiment(
    input: &Path,
    output: &Path,
    experiments: u32,
    graph_points: u32,
    opts: ExperimentOpts,
) {
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

    let rot_len = opts.shape.iter().product::<u32>();
    let logger = Rc::new(Cell::new(Some(crate::agent::Logger::new(
        &output.join("regret.csv"),
        experiments * opts.trials * rot_len / graph_points,
    ))));

    let videos = load_data(input.as_ref())
        .map(|(path, thumbs)| {
            let agent = crate::agent::ThompsonSampler::new(thumbs.len(), opts.alpha, opts.beta);
            Video::new(
                thumbs,
                crate::agent::RegretLogger::new(agent, logger.clone()),
                path,
            )
        })
        .collect();

    let clickability = clickability(&opts.shape);
    let mut experiment = Experiment::new(videos, clickability, opts);

    for e in 0..experiments {
        experiment.run();
        let img = rotator_snapshot(&mut experiment);
        img.save(output.join(format!("thumbnails-{:02}.png", e)))
            .expect("Failed to write thumbnails");
        println!("Finished experiment {}", e);
    }
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

pub fn load_data(input: &Path) -> impl Iterator<Item = (PathBuf, Vec<Thumbnail>)> {
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

                        Thumbnail::new(sub[0].parse().unwrap(), sub[1].parse().unwrap())
                    })
                    .collect::<Vec<_>>();
            (path, thumbs)
        })
}

pub fn clickability(shape: &[u32; 2]) -> Vec<f32> {
    let mat_len = shape.iter().product::<u32>() as usize;
    (0..mat_len).map(|i| f32::powf(0.99, i as f32)).collect()
}

fn rotator_snapshot<A: Agent>(experiment: &mut Experiment<A>) -> image::RgbImage {
    let impressions = experiment.generate_impressions();

    let sample_thumb = impressions[0].0;
    let sample_thumb =
        image::open(experiment.videos()[sample_thumb.video()].thumb_path(sample_thumb))
            .expect("Failed to read image");
    let (img_width, img_height) = image::GenericImageView::dimensions(&sample_thumb);
    let [width, height] = experiment.shape.clone();

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
