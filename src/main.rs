use image::GenericImageView;
use rand::{seq::IteratorRandom, Rng, SeedableRng};
use rand_distr::{Beta, Distribution};
use std::{
    cell::Cell,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    rc::Rc,
};

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
    /// Get the expected optimal reward
    fn optimal(&self) -> Reward;
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

    /// The mean of the distribution
    fn mean(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
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
#[derive(Debug)]
struct ThompsonSampler {
    /// The params for the distributions
    dist_params: Vec<BetaParams>,
    /// The distributions used to choose actions
    dist: Vec<Beta<f32>>,
    optimal: (usize, f32),
}

impl ThompsonSampler {
    /// Construct a new `ThompsonSampler`
    fn new(arms: usize) -> Self {
        Self {
            dist_params: vec![BetaParams::default(); arms],
            dist: vec![BetaParams::default().new_dist(); arms],
            optimal: (0, BetaParams::default().mean()),
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

struct RegretLogger<A> {
    agent: A,
    max: f32,
    file: Rc<Cell<Option<BufWriter<File>>>>,
}

impl<A: std::fmt::Debug> std::fmt::Debug for RegretLogger<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegretLogger")
            .field("agent", &self.agent)
            .field("max", &self.max)
            .finish()
    }
}

impl<A> RegretLogger<A> {
    fn new(agent: A, file: Rc<Cell<Option<BufWriter<File>>>>) -> Self {
        Self {
            agent,
            max: 0.,
            file,
        }
    }
}

impl<A: Agent> Agent for RegretLogger<A> {
    fn choose<G: rand::Rng>(&mut self, rng: &mut G) -> Action {
        self.agent.choose(rng)
    }

    fn update(&mut self, a: Action, r: Reward) {
        self.agent.update(a, r);

        let regret = self.agent.optimal().0 - r.0;
        let mut file = self.file.take().expect("File is in use");
        writeln!(&mut file, "{}", regret).expect("Failed to write to file");
        self.file.set(Some(file));
    }

    fn optimal(&self) -> Reward {
        self.agent.optimal()
    }
}

/// The data of a thumbnail
#[derive(Debug)]
struct Thumbnail {
    /// The id of the thumbnail
    id: u32,
    /// Click through ratio
    ctr: f64,
    /// The number of clicks that the thumbnail has received
    clicks: u32,
    /// The number of impressions that the thumbnail has received
    impressions: u32,
}

impl Thumbnail {
    /// Register an imression
    fn impress(&mut self, click: bool) {
        self.impressions += 1;
        self.clicks += click as u32;
    }
}

#[derive(Debug)]
struct Video {
    thumbs: Vec<Thumbnail>,
    agent: RegretLogger<ThompsonSampler>,
    path: PathBuf,
}

impl Video {
    fn thumb_path(&self, id: ThumbId) -> PathBuf {
        self.path.join(format!("{:02}.png", self.thumbs[id.1].id))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct VideoId(usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThumbId(VideoId, usize);

struct Experiment {
    videos: Vec<Video>,
    trials: u32,
    shape: [u32; 2],
    rng: DefaultRng,
}

impl Experiment {
    fn update_thumb(&mut self, id: ThumbId, click: bool) {
        let reward = if click { 1. } else { 0. };
        let vid = &mut self.videos[id.0 .0];
        vid.thumbs[id.1].impress(click);
        vid.agent.update(Action(id.1), Reward(reward));
    }

    fn update(&mut self, mut impressions: Vec<ThumbId>, mut clicks: Vec<ThumbId>) {
        impressions.sort_unstable_by_key(|ThumbId(vid, _)| *vid);
        clicks.sort_unstable_by_key(|ThumbId(vid, _)| *vid);
        let mut c = 0;
        let mut i = 0;

        // Separate impressions which lead to clicks from these that did not
        loop {
            let (&imp, &click) = match (impressions.get(i), clicks.get(c)) {
                // We received a click that didn't have an impression
                (None, Some(_)) => panic!("Click not contained in impressions"),
                // We ran out of clicks, so lets speed through the remaining imressions
                (Some(_), None) => {
                    for &id in &impressions[i..] {
                        self.update_thumb(id, false);
                    }
                    break;
                }
                // Got a click and an impression, proceed to check what to do
                (Some(imp), Some(click)) => (imp, click),
                // All done
                (None, None) => break,
            };

            match imp.0.cmp(&click.0) {
                // Not a click
                std::cmp::Ordering::Less => {
                    self.update_thumb(imp, false);
                    i += 1;
                }
                // Click
                std::cmp::Ordering::Equal => {
                    self.update_thumb(click, true);
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

    fn generate_impressions(&mut self) -> (Vec<ThumbId>, Vec<(ThumbId, f64)>) {
        let n = self.shape.iter().product::<u32>() as usize;
        // Pick the videos that will be displayed
        self.videos
            .iter_mut()
            .enumerate()
            .choose_multiple(&mut self.rng, n)
            .into_iter()
            .map(|(id, vid)| {
                let thumb = vid.agent.choose(&mut self.rng).0;
                let ctr = vid.thumbs[thumb].ctr;
                (ThumbId(VideoId(id), thumb), ctr)
            })
            .fold(
                (Vec::new(), Vec::new()),
                |(mut impress, mut probs), (id, ctr)| {
                    impress.push(id);
                    probs.push((id, ctr));
                    (impress, probs)
                },
            )
    }

    fn run_trial(&mut self) {
        let (impressions, probs) = self.generate_impressions();
        let clicks = probs
            .into_iter()
            .filter_map(|(id, prob)| self.rng.gen_bool(prob).then_some(id))
            .collect();

        self.update(impressions, clicks)
    }

    fn run(&mut self) {
        for _ in 0..self.trials {
            self.run_trial()
        }
    }
}

/// A seed to initialize random behaviour
const SEED: u64 = 0;
const SHAPE: [u32; 2] = [5, 10];
const TRIALS: u32 = 1000;
const EXPERIMENTS: u32 = 100;

fn main() {
    // The input data folder
    let input = std::env::args().nth(1).expect("Expected an input argument");
    // The output data folder
    let output = std::env::args()
        .nth(2)
        .expect("Expected an output argument")
        .parse::<PathBuf>()
        .unwrap();

    if output.exists() && output.is_relative() {
        std::fs::remove_dir_all(&output).expect("Failed to clean output dir")
    }
    std::fs::create_dir(&output).expect("Failed to create output directory");

    let regret = File::options()
        .create(true)
        .write(true)
        .open(output.join("regret.csv"))
        .expect("Failed to open regret file");
    let regret = Rc::new(Cell::new(Some(BufWriter::new(regret))));

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

                    Thumbnail {
                        id: sub[0].parse().unwrap(),
                        ctr: sub[1].parse().unwrap(),
                        clicks: 0,
                        impressions: 0,
                    }
                })
                .collect::<Vec<_>>();

            let agent = ThompsonSampler::new(thumbs.len());

            Video {
                thumbs,
                agent: RegretLogger::new(agent, regret.clone()),
                path,
            }
        })
        .collect::<Vec<_>>();

    let rng = DefaultRng::seed_from_u64(SEED);

    let mut experiment = Experiment {
        videos,
        trials: TRIALS,
        shape: SHAPE,
        rng,
    };

    for e in 0..EXPERIMENTS {
        experiment.run();
        let (impressions, _) = experiment.generate_impressions();

        let sample_thumb = impressions[0];
        let sample_thumb =
            image::open(experiment.videos[sample_thumb.0 .0].thumb_path(sample_thumb))
                .expect("Failed to read image");
        let (img_width, img_height) = sample_thumb.dimensions();
        let [width, height] = SHAPE;

        let mut image = image::RgbImage::new(width * img_width, height * img_height);

        for y in 0..height {
            for x in 0..width {
                let n = y * width + x;
                let id = impressions[n as usize];
                let path = experiment.videos[id.0 .0].thumb_path(id);
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
