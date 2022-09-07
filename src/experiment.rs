use super::agent::{Action, Agent, RegretLogger, Reward, ThompsonSampler};
use super::DefaultRng;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng, SeedableRng,
};
use std::path::PathBuf;

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

    pub fn replace_agent(&mut self, agent: A) -> A {
        std::mem::replace(&mut self.agent, agent)
    }

    pub fn thumbs(&self) -> &[Thumbnail] {
        self.thumbs.as_ref()
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
    clicks: u32,
    impressions: u32,
    ratio: f32,
}

impl Default for Clickability {
    fn default() -> Self {
        Self {
            clicks: 0,
            impressions: 0,
            ratio: 0.,
        }
    }
}

impl Clickability {
    fn click(&mut self, click: bool) {
        self.clicks += click as u32;
        self.impressions += 1;
        self.ratio = self.clicks as f32 / self.impressions as f32;
    }

    pub fn ratio(&self) -> f32 {
        self.ratio.max(0.)
    }
}

pub struct Experiment<A> {
    // Video data
    videos: Vec<Video<A>>,
    // Shape of the rotator
    shape: [u32; 2],
    // Number of trials per experiment
    trials: u32,
    // The real clickability matrix
    clickability: Vec<f32>,
    // Estimate of the clickability matrix
    click_estimate: Vec<Clickability>,
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
    pub fn new(
        videos: Vec<Video<A>>,
        shape: [u32; 2],
        trials: u32,
        clickability: Vec<f32>,
        test_prob: f64,
        seed: u64,
    ) -> Self {
        Self {
            videos,
            shape,
            trials,
            clickability,
            click_estimate: vec![Default::default(); shape.iter().product::<u32>() as usize],
            test_prob,
            reward: 0.,
            rng: DefaultRng::seed_from_u64(seed),
        }
    }

    fn update_thumb(&mut self, thumb: ThumbId, cell: CellId, click: bool, test: bool) {
        // Update clickability
        if test {
            self.click_estimate[cell.0].click(click);
        }

        // Update thumbnail data
        let vid = &mut self.videos[thumb.0 .0];
        vid.thumbs[thumb.1].impress(click);

        // Update agent
        let reward = if click { 1. } else { 0. };
        vid.agent.update(Action(thumb.1), Reward(reward));
        self.reward += reward;
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

    pub fn generate_impressions(
        &mut self,
    ) -> (Vec<(ThumbId, CellId)>, Vec<((ThumbId, CellId), f64)>) {
        let n = self.shape.iter().product::<u32>() as usize;

        // Pick the videos that will be displayed
        let mut videos = self
            .videos
            .iter_mut()
            .enumerate()
            .choose_multiple(&mut self.rng, n);
        videos.shuffle(&mut self.rng);

        let (_, impressions, probs) = videos
            .into_iter()
            .map(|(id, vid)| {
                let thumb = vid.agent.choose(&mut self.rng).0;
                let ctr = vid.thumbs[thumb].ctr;
                (ThumbId(VideoId(id), thumb), ctr)
            })
            .fold(
                (0, Vec::new(), Vec::new()),
                |(cell, mut impress, mut probs), (id, ctr)| {
                    let imp = (id, CellId(cell));
                    impress.push(imp);
                    probs.push((imp, ctr));
                    (cell + 1, impress, probs)
                },
            );
        (impressions, probs)
    }

    fn run_trial(&mut self) {
        let (impressions, probs) = self.generate_impressions();
        let clicks = probs
            .into_iter()
            .filter_map(|((thumb, cell), prob)| {
                self.rng
                    .gen_bool(prob * self.clickability[cell.0] as f64)
                    .then_some((thumb, cell))
            })
            .collect();

        let test = self.rng.gen_bool(self.test_prob);

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

    pub fn videos_mut(&mut self) -> &mut Vec<Video<A>> {
        &mut self.videos
    }

    pub fn clickability(&self) -> &[f32] {
        self.clickability.as_ref()
    }

    pub fn click_estimate(&self) -> &[Clickability] {
        self.click_estimate.as_ref()
    }

    pub fn reward(&self) -> f32 {
        self.reward
    }
}
