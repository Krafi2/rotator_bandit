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

/// The reward returned by the environment. Either `1` or `0`, but with a weight.
#[derive(Debug, Clone, Copy)]
pub struct Reward(pub bool, pub f32);

/// A common trait for all agents
pub trait Agent {
    /// Choose an action
    fn choose(&self, rng: &mut DefaultRng) -> Action;
    /// Update the agent with the result of an action
    fn update(&mut self, a: Action, r: Reward);
    /// Get the expected optimal reward
    fn optimal(&self) -> f32;
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

    /// Update the distribution to better match the reward
    pub fn update(&mut self, reward: Reward) {
        let Reward(reward, weight) = reward;
        let reward = reward as u32 as f32;
        let a = weight * reward;
        self.alpha += a;
        self.beta += weight - a; // weight * (1. - reward)
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

/// An agent which uses beta thompson sampling to select actions
#[derive(Debug, Clone)]
pub struct ThompsonSampler {
    /// The params for the distributions
    dist_params: Vec<BetaParams>,
    /// The distributions used to choose actions
    dist: Vec<Beta<f32>>,
    /// The index of the arm with the best mean reward, and the mean itself
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
        // Choose the action with the highest randomly sampled value
        let a = arg_max(self.dist.iter().map(|dist| dist.sample(rng)));
        Action(a)
    }

    /// Update the distribution and optimal reward
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

    /// The mean of the best performing arm
    fn optimal(&self) -> f32 {
        self.optimal.1
    }
}

/// A shared logger for writing regret data to a file
pub struct Logger {
    /// Accumulated regret
    accumulator: f32,
    /// The number of impressions
    n: u32,
    /// The number of samples to accumulate before writing to file
    samples: u32,
    /// The file writer
    file: BufWriter<File>,
    /// Reward accumulator
    reward: f32,
}

impl Logger {
    /// Construct a new `Logger`
    pub fn new(path: &Path, samples: u32) -> Self {
        let file = File::options()
            .create(true)
            .write(true)
            .truncate(true)
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

    /// Update the logger with new data
    fn update(&mut self, reward: f32, regret: f32) {
        // Write the accumulated samples to the file
        if self.n % self.samples == 0 {
            writeln!(&mut self.file, "{}, {}", self.n, self.accumulator)
                .expect("Failed to write to file");
        }
        self.reward += reward;
        self.accumulator += regret;
        self.n += 1;
    }

    /// Flush the writer
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }

    /// Get the accumulated reward
    pub fn reward(&self) -> f32 {
        self.reward
    }
}

/// A wrapper around an agent that writes regret to a shared logger
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
    /// Construct a new `RegretLogger`
    pub fn new(agent: A, logger: Rc<Cell<Option<Logger>>>) -> Self {
        Self { agent, logger }
    }
}

impl<A: Agent> Agent for RegretLogger<A> {
    fn choose(&self, rng: &mut DefaultRng) -> Action {
        // Delegate the choice to the agent
        self.agent.choose(rng)
    }

    /// Update the underlying agent and the logger
    fn update(&mut self, a: Action, r: Reward) {
        self.agent.update(a, r);
        let regret = self.agent.optimal() - r.1;
        let mut logger = self.logger.take().expect("File is in use");
        logger.update(r.1, regret);
        self.logger.set(Some(logger));
    }

    fn optimal(&self) -> f32 {
        // Delegate the the agent
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
