mod agent;
mod experiment;
mod optimize;

use experiment::ExperimentOpts;
use optimize::OptimizeOpts;
use std::path::PathBuf;

pub type DefaultRng = rand_chacha::ChaCha8Rng;

/// A seed to initialize random behaviour
const SEED: u64 = 0;
/// The shape of the rotator
const SHAPE: [u32; 2] = [5, 10];
/// The number of trials per experiment
const TRIALS: u32 = 10_000;
/// The number of experiments
const EXPERIMENTS: u32 = 1;
/// How many points should be plotted to the graph
const GRAPH_POINTS: u32 = 600;
/// Probability of a trial being a test. Currently there is no reason to set it lower than 1 since
/// all trials sample data randomly based on the probabilities in the dataset.
const TEST_PROB: f64 = 1.;

/// The domain of alpha and beta params to search
const DOMAIN: [std::ops::Range<f32>; 2] = [0.001..100., 0.001..100.];
/// Width of the grid
const WIDTH: usize = 64;
/// Height of the grid
const HEIGHT: usize = 64;
/// Number of samples per pixel. The experiments are a bit noisy so try to increase this to
/// decrease noise at the cost of longer run time.
const SAMPLES: u32 = 1;

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
            "experiment" => Op::Experiment,
            other => panic!("Unrecognized operation '{}'", other),
        })
        .expect("Expected operation");

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

    let eopts = ExperimentOpts {
        seed: SEED,
        shape: SHAPE,
        trials: TRIALS,
        test_prob: TEST_PROB,
    };

    let opts = OptimizeOpts {
        domain: DOMAIN,
        width: WIDTH,
        height: HEIGHT,
        samples: SAMPLES,
    };

    match op {
        Op::Optimize => optimize::optimize(&input, &output, opts, eopts),
        Op::Experiment => experiment::experiment(&input, &output, EXPERIMENTS, GRAPH_POINTS, eopts),
    }
}
