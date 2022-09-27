use crate::DefaultRng;
use rand_distr::{Beta, Distribution};
use std::{
    cell::Cell,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    rc::Rc,
};

/// An action chosen by an agent
#[derive(Debug, Clone, Copy)]
pub struct Action(pub usize);

/// The reward returned by the environment
#[derive(Debug, Clone, Copy)]
pub struct Reward(pub f32);

/// A common trait for all agents
pub trait Agent {
    /// Choose an action
    fn choose(&self, rng: &mut DefaultRng) -> Action;
    /// Update the agent with the result of an action
    fn update(&mut self, a: Action, r: Reward);
    /// Get the expected optimal reward
    fn optimal(&self) -> Reward;
}

/// Parameters for a beta distribution
#[derive(Debug, Clone)]
pub struct BetaParams {
    pub alpha: f32,
    pub beta: f32,
}

impl BetaParams {
    /// Construct a new distribution from the parameters
    pub fn new_dist(&self) -> Beta<f32> {
        match Beta::new(self.alpha, self.beta) {
            Err(_) => panic!(
                "Invalid parameters for beta distribution (alpha: {}, beta: {})",
                self.alpha, self.beta
            ),
            Ok(dist) => dist,
        }
    }

    /// The mean of the distribution
    pub fn mean(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
    }

    pub fn update(&mut self, reward: Reward) {
        let r = reward.0;
        self.alpha += r;
        self.beta += 1. - r;
    }
}

impl Default for BetaParams {
    /// A default beta distribution with alpha and beta of 1.
    fn default() -> Self {
        Self {
            alpha: 1.,
            beta: 1.,
        }
    }
}

/// An agent which uses beta thompson sampling to
#[derive(Debug, Clone)]
pub struct ThompsonSampler {
    /// The params for the distributions
    dist_params: Vec<BetaParams>,
    /// The distributions used to choose actions
    dist: Vec<Beta<f32>>,
    optimal: (usize, f32),
}

impl ThompsonSampler {
    /// Construct a new `ThompsonSampler`
    pub fn new(arms: usize, alpha: f32, beta: f32) -> Self {
        let params = BetaParams { alpha, beta };
        Self {
            dist_params: vec![params.clone(); arms],
            dist: vec![params.new_dist(); arms],
            optimal: (0, params.mean()),
        }
    }
}

impl Agent for ThompsonSampler {
    fn choose(&self, rng: &mut DefaultRng) -> Action {
        let a = arg_max(self.dist.iter().map(|dist| dist.sample(rng)));
        Action(a)
    }

    fn update(&mut self, a: Action, r: Reward) {
        let a = a.0;
        let dist = &mut self.dist_params[a];
        // Update the params
        dist.update(r);
        let (arm, opt) = self.optimal;
        let mean = dist.mean();
        if a == arm || mean > opt {
            self.optimal = (a, mean);
        }
        // Update the distribution
        self.dist[a] = dist.new_dist();
    }

    fn optimal(&self) -> Reward {
        Reward(self.optimal.1)
    }
}

pub struct Logger {
    accumulator: f32,
    n: u32,
    samples: u32,
    file: BufWriter<File>,
    reward: f32,
}

impl Logger {
    pub fn new(path: &Path, samples: u32) -> Self {
        let file = File::options()
            .create(true)
            .write(true)
            .open(path)
            .expect("Failed to open regret file");

        Self {
            accumulator: 0.,
            n: 0,
            samples: samples.max(1),
            file: BufWriter::new(file),
            reward: 0.,
        }
    }

    fn update(&mut self, reward: f32, regret: f32) {
        if self.n % self.samples == 0 {
            writeln!(&mut self.file, "{}, {}", self.n, self.accumulator)
                .expect("Failed to write to file");
        }
        self.reward += reward;
        self.accumulator += regret;
        self.n += 1;
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }

    pub fn reward(&self) -> f32 {
        self.reward
    }
}

pub struct RegretLogger<A> {
    agent: A,
    logger: Rc<Cell<Option<Logger>>>,
}

impl<A: std::fmt::Debug> std::fmt::Debug for RegretLogger<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegretLogger")
            .field("agent", &self.agent)
            .finish()
    }
}

impl<A> RegretLogger<A> {
    pub fn new(agent: A, logger: Rc<Cell<Option<Logger>>>) -> Self {
        Self { agent, logger }
    }
}

impl<A: Agent> Agent for RegretLogger<A> {
    fn choose(&self, rng: &mut DefaultRng) -> Action {
        self.agent.choose(rng)
    }

    fn update(&mut self, a: Action, r: Reward) {
        self.agent.update(a, r);
        let regret = self.agent.optimal().0 - r.0;
        let mut logger = self.logger.take().expect("File is in use");
        logger.update(r.0, regret);
        self.logger.set(Some(logger));
    }

    fn optimal(&self) -> Reward {
        self.agent.optimal()
    }
}

/// Get the argmax of a collection
fn arg_max<I: IntoIterator>(collection: I) -> usize
where
    I::Item: std::cmp::PartialOrd,
{
    collection
        .into_iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Comparison failed"))
        .expect("Expected at least one element")
        .0
}
