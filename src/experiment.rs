use std::path::PathBuf;

use crate::agent::BetaParams;

use super::agent::{Action, Agent, Reward};
use super::DefaultRng;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng, SeedableRng,
};
use rand_distr::Distribution;

/// The data of a thumbnail
#[derive(Debug, Clone)]
pub struct Thumb {
    /// The id of the thumbnail
    pub id: u32,
    /// Click through ratio
    pub ctr: f64,
    /// The number of clicks that the thumbnail has received
    pub clicks: u32,
    /// The number of impressions that the thumbnail has received
    pub impressions: u32,
}

impl Thumb {
    /// Construct a new `Thumb`
    pub fn new(id: u32, ctr: f64) -> Self {
        Self {
            id,
            ctr,
            clicks: 0,
            impressions: 0,
        }
    }

    /// Register an impression
    fn impress(&mut self, click: bool) {
        self.impressions += 1;
        self.clicks += click as u32;
    }

    /// Reset the data
    fn reset(&mut self) {
        self.clicks = 0;
        self.impressions = 0;
    }
}

/// The data of a video
pub struct Video<A> {
    /// Thumbnails
    thumbs: Vec<Thumb>,
    /// Agent for selecting thumbnails
    agent: A,
    /// The path to the video data file
    path: PathBuf,
}

impl<A> Video<A> {
    pub fn new(thumbs: Vec<Thumb>, agent: A, path: PathBuf) -> Self {
        Self {
            thumbs,
            agent,
            path,
        }
    }

    /// Get the path to a specific thumbnail
    pub fn thumb_path(&self, id: ThumbId) -> PathBuf {
        self.path.join(format!("{:02}.png", self.thumbs[id.1].id))
    }
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

/// Id of a `[Video]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct VideoId(usize);

/// Id of a `[Video]`'s `[Thumbnail]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThumbId(VideoId, usize);

impl ThumbId {
    pub fn video(&self) -> usize {
        self.0 .0
    }
}

/// Id of a cell within the rotator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellId(pub usize);

/// Clickability data of a cell withing the rotator
#[derive(Debug, Clone)]
pub struct Clickability {
    /// Number of clicks
    clicks: Vec<u32>,
    /// Number of impressions
    impressions: Vec<u32>,
    /// Clickability ration of the first cell. This is used to normalize the clickability matrix.
    scale: f32,
}

const CLICK_BIAS: u32 = 1;
impl Clickability {
    fn new(n: usize) -> Self {
        Self {
            clicks: vec![CLICK_BIAS; n],
            impressions: vec![CLICK_BIAS; n],
            scale: 1.,
        }
    }

    /// Update the clickability of a cell with an impression
    fn update(&mut self, cell: CellId, click: bool) {
        let id = cell.0;
        self.clicks[id] += click as u32;
        self.impressions[id] += 1;
        // If the target is the first cell, update the scale factor as well
        if id == 0 {
            self.scale = self.clicks[id] as f32 / self.impressions[id] as f32;
        }
    }

    /// Get the weight of a cell. Note that this is the reciprocical of the clickability.
    pub fn weight(&self, cell: CellId) -> f32 {
        let id = cell.0;
        self.impressions[id] as f32 / self.clicks[id] as f32 * self.scale
    }

    // Reset the data to the original state
    fn reset(&mut self) {
        for clicks in &mut self.clicks {
            *clicks = CLICK_BIAS;
        }
        for imp in &mut self.impressions {
            *imp = CLICK_BIAS;
        }
        self.scale = 1.;
    }
}

/// The "ratings" of the videos. This is used for populating the rotator in a way that places the
/// best videos up front.
#[derive(Debug, Clone)]
struct Ratings {
    alpha: f32,
    beta: f32,
    dist_params: Vec<BetaParams>,
    dist: Vec<rand_distr::Beta<f32>>,
}

impl Ratings {
    /// Construct a new `Ratings`
    fn new(n: usize, alpha: f32, beta: f32) -> Self {
        let params = BetaParams { alpha, beta };
        Self {
            alpha,
            beta,
            dist_params: vec![params.clone(); n],
            dist: vec![params.new_dist(); n],
        }
    }

    /// Reset the data to the initial state
    fn reset(&mut self) {
        *self = Self::new(self.dist.len(), self.alpha, self.beta)
    }

    /// Update the rating of a video with a reward
    fn update(&mut self, id: VideoId, reward: Reward) {
        let dist = &mut self.dist_params[id.0];
        dist.update(reward);
        self.dist[id.0] = dist.new_dist();
    }

    /// Choose a list of videos to populate the rotator
    fn choose(&self, n: usize, rng: &mut DefaultRng) -> Vec<VideoId> {
        // This works essentially the same way as a normal Thompson sampler, but we take the `n`
        // best value as opposed to a single one.
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

/// Experiment options
pub struct ExperimentOpts {
    pub seed: u64,
    pub shape: [u32; 2],
    pub trials: u32,
    pub test_prob: f64,
    pub alpha: f32,
    pub beta: f32,
}

/// The data of an experiment
pub struct Experiment<A> {
    /// Video ratings
    ratings: Ratings,
    /// Video data
    videos: Vec<Video<A>>,
    /// Shape of the rotator
    shape: [u32; 2],
    /// Number of trials per experiment
    trials: u32,
    /// The real clickability matrix
    clickability: Vec<f32>,
    /// Estimate of the clickability matrix
    click_estimate: Clickability,
    /// Probabity of a test trial
    test_prob: f64,
    /// The accumulated reward
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
    /// Construct a new `Experiment`
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

    /// Update a thumbnail with an impression. This also updates all the underlying mechanisms
    /// which have to keep track of rewards.
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

    /// Receive a list of impressions and clicks and perform the neccessary updates
    fn update(
        &mut self,
        // A list of impressions
        mut impressions: Vec<(ThumbId, CellId)>,
        // A list of clicks
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
                // Probably a bug. There was a click that had no associated impression.
                std::cmp::Ordering::Greater => {
                    panic!("Click not contained in impressions")
                }
            }
        }
    }

    /// Generate random impressions from the rotator for testing
    pub fn generate_test_impressions(&mut self) -> Vec<(ThumbId, CellId)> {
        let n = self.shape.iter().product::<u32>() as usize;
        // Randomly pick the videos that will be displayed
        let mut videos = self
            .videos
            .iter_mut()
            .enumerate()
            .choose_multiple(&mut self.rng, n);
        videos.shuffle(&mut self.rng);

        // Pick thumbnails to go with the videos
        videos
            .into_iter()
            .enumerate()
            .map(|(cell, (id, vid))| {
                let thumb = vid.agent.choose(&mut self.rng).0;
                (ThumbId(VideoId(id), thumb), CellId(cell))
            })
            .collect()
    }

    /// Generate impressions from the rotator
    pub fn generate_impressions(&mut self) -> Vec<(ThumbId, CellId)> {
        let n = self.shape.iter().product::<u32>() as usize;
        // Pick the videos and thumbnails that will be displayed
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

    /// Run a trial
    fn run_trial(&mut self) {
        // Generate impressions
        let test = self.rng.gen_bool(self.test_prob);
        let impressions = if test {
            self.generate_test_impressions()
        } else {
            self.generate_impressions()
        };

        // Generate clicks from the impressions
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

    /// Run the experiment
    pub fn run(&mut self) {
        for _ in 0..self.trials {
            self.run_trial()
        }
    }

    /// Get the videos
    pub fn videos(&self) -> &[Video<A>] {
        self.videos.as_ref()
    }

    /// Get the clickability
    pub fn clickability(&self) -> &[f32] {
        self.clickability.as_ref()
    }

    /// Get the clickability estimate
    pub fn click_estimate(&self) -> &Clickability {
        &self.click_estimate
    }

    /// Get the accumulated reward
    pub fn reward(&self) -> f32 {
        self.reward
    }

    /// Set the test probability
    pub fn set_testp(&mut self, prob: f64) {
        self.test_prob = prob;
    }

    /// Get the rotator's shape
    pub fn shape(&self) -> &[u32; 2] {
        &self.shape
    }
}

impl<A: Clone> Experiment<A> {
    /// Reset the experiment and replace the agents with clones of `agent`
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
