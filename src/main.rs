use rand::{Rng, SeedableRng};
use rand_distr::{Beta, Distribution};
use std::io::BufRead;

type DefaultRng = rand_chacha::ChaCha8Rng;

/// An action chosen by an agent
#[derive(Debug, Clone, Copy)]
struct Action(usize);

/// The reward returned by the environment
#[derive(Debug, Clone, Copy)]
struct Reward(f32);

/// A common trait for all bandits
trait Bandit {
    /// Pull one of the bandit's arms and receive a reward
    fn pull(&mut self, arm: Action) -> Reward;
}

/// A common trait for all agents
trait Agent {
    /// Choose an action
    fn choose<G: rand::Rng>(&mut self, rng: &mut G) -> Action;
    /// Update the agent with the result of an action
    fn update(&mut self, a: Action, r: Reward);
}

/// Parameters for a beta distribution
#[derive(Debug, Clone)]
struct BetaParams {
    alpha: f32,
    beta: f32,
}

impl BetaParams {
    /// Construct a new distribution from the parameters
    fn new_dist(&self) -> Beta<f32> {
        Beta::new(self.alpha, self.beta).expect("Invalid parameters for beta distribution")
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
struct ThompsonSampler {
    /// The params for the distributions
    dist_params: Vec<BetaParams>,
    /// The distributions used to choose actions
    dist: Vec<Beta<f32>>,
}

impl ThompsonSampler {
    /// Construct a new `ThompsonSampler`
    fn new(arms: usize) -> Self {
        Self {
            dist_params: vec![BetaParams::default(); arms],
            dist: vec![BetaParams::default().new_dist(); arms],
        }
    }
}

impl Agent for ThompsonSampler {
    fn choose<G: rand::Rng>(&mut self, rng: &mut G) -> Action {
        let a = arg_max(self.dist.iter().map(|dist| dist.sample(rng)));
        Action(a)
    }

    fn update(&mut self, a: Action, r: Reward) {
        let a = a.0;
        let r = r.0;
        let dist = &mut self.dist_params[a];
        // Update the params
        dist.alpha += r;
        dist.beta += 1. - r;
        // Update the distribution
        self.dist[a] = dist.new_dist();
    }
}

/// A bandit for testing purposes which uses pregenerated data
struct TestBandit {
    /// The thumbnail data
    data: Vec<Thumbnail>,
    /// An rng for simulating user interactions
    rng: DefaultRng,
}

impl TestBandit {
    /// Construct a new `TestBandit`
    fn new(data: Vec<Thumbnail>, seed: u64) -> Self {
        Self {
            data,
            rng: DefaultRng::seed_from_u64(seed),
        }
    }
}

impl Bandit for TestBandit {
    fn pull(&mut self, arm: Action) -> Reward {
        let thumb = &self.data[arm.0];
        if self.rng.gen_bool(thumb.ctr) {
            // No click
            Reward(0.)
        } else {
            // Click
            Reward(1.)
        }
    }
}

#[allow(dead_code)]
/// The data of a thumbnail
struct Thumbnail {
    /// The id of the thumbnail
    id: u32,
    /// Click through ratio
    ctr: f64,
    /// The number of clicks that the thumbnail received
    clicks: u32,
    /// The number of impressions that the thumbnail received
    impressions: u32,
}

/// A seed to initialize random behaviour
const SEED: u64 = 0;

fn main() {
    // The input data file
    let input = std::env::args().nth(1).expect("Expected an input argument");
    // How many impressions should be simulated for testing
    let impressions = std::env::args()
        .nth(2)
        .expect("Expected an impressions argument")
        .parse::<usize>()
        .expect("Expected a number");

    // Parse the thumbnail data from the csv file
    let thumbnails =
        std::io::BufReader::new(std::fs::File::open(input).expect("Failed to read input"))
            .lines()
            .map(|s| {
                let s = s.expect("Failed to read line");
                let sub = s.split(',').collect::<Vec<_>>();

                Thumbnail {
                    id: sub[0].parse().unwrap(),
                    ctr: sub[1].parse().unwrap(),
                    clicks: sub[2].parse().unwrap(),
                    impressions: sub[3].parse().unwrap(),
                }
            })
            .collect::<Vec<_>>();

    let mut agent = ThompsonSampler::new(thumbnails.len());
    let mut bandit = TestBandit::new(thumbnails, SEED);
    let mut rng = DefaultRng::seed_from_u64(SEED);

    let mut reward = 0.;
    for _ in 0..impressions {
        let a = agent.choose(&mut rng);
        let r = bandit.pull(a);
        agent.update(a, r);
        reward += r.0;
    }

    println!(
        "Accumulated a reward of {} over the course of {} impressions!",
        reward, impressions
    );
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
