use distrs::Normal;
use friedrich::{kernel::Gaussian, prior::ConstantPrior};

use crate::agent::{Agent, ThompsonSampler};

type GaussianProcess = friedrich::gaussian_process::GaussianProcess<Gaussian, ConstantPrior>;
type Experiment = crate::experiment::Experiment<ThompsonSampler>;

#[derive(Debug, Clone)]
pub struct Point(Vec<f64>);

#[derive(Debug, Clone)]
pub struct Sample {
    point: Point,
    val: f64,
}

impl Sample {
    pub fn new(point: Point, val: f64) -> Self {
        Self { point, val }
    }
}

pub struct Optimizer {
    experiment: Experiment,
    initial_samples: u32,
    samples: u32,
    pick: u32,
    refine: u32,
    domain: Vec<std::ops::Range<f64>>,
    surrogate: GaussianProcess,
    best: Sample,
    seed: u64,
}

impl Optimizer {
    fn eval(&mut self, point: &Point) -> f64 {
        let mut experiment = self.experiment.clone();
        let [alpha, beta]: [f64; 2] = point.0.as_slice().try_into().unwrap();
        for vid in experiment.videos_mut() {
            let arms = vid.thumbs().len();
            vid.replace_agent(ThompsonSampler::new(arms, alpha as f32, beta as f32));
        }
        experiment.run();
        experiment.reward() as f64
    }

    fn acquire(&mut self) -> Point {
        grid_search(
            &self.domain,
            self.samples,
            self.pick,
            self.refine,
            |point| expected_improvement(&self.surrogate, point, self.best.val),
        )
    }

    pub fn next_sample(&mut self) -> Sample {
        let point = self.acquire();
        let val = self.eval(&point);
        let sample = Sample::new(point, val);

        if val > self.best.val {
            self.best = sample.clone()
        }
        sample
    }
}

// Expected improvement acquisition function as described [here](https://ash-aldujaili.github.io/blog/2018/02/01/ei/).
fn expected_improvement(surrogate: &GaussianProcess, point: &Vec<f64>, best: f64) -> f64 {
    let variance = surrogate.predict_variance(point);
    let deviation = variance.sqrt();
    let mean = surrogate.predict(point);
    let a = best - mean;
    let b = a / deviation;
    let cdf = Normal::cdf(b, 0., 1.);
    let pdf = Normal::pdf(b, 0., 1.);
    a * cdf + deviation * pdf
}

struct Sector {
    pos: Vec<u32>,
    val: f64,
}

fn grid_search<F: Fn(&Vec<f64>) -> f64>(
    domain: &[std::ops::Range<f64>],
    samples: u32,
    pick: u32,
    refine: u32,
    func: F,
) -> Point {
    let d = domain.len();

    let origin = domain.iter().map(|range| range.start).collect::<Vec<_>>();

    let mut steps = domain
        .iter()
        .map(|range| (range.end - range.start) / samples as f64)
        .collect::<Vec<_>>();

    let mut sectors = vec![Sector {
        pos: vec![0; d],
        val: func(&origin),
    }];

    let mut new_sectors = Vec::new();

    for _ in 0..refine {
        sectors.sort_unstable_by(|a, b| a.val.partial_cmp(&b.val).unwrap().reverse());
        for sector in &sectors[..pick as usize] {
            refine_sector(sector, &steps, samples, &mut new_sectors, |pos| {
                let point = place_point(pos, &steps, &origin);
                func(&point)
            })
        }

        let scale = 1. / samples as f64;
        for s in &mut steps {
            *s *= scale;
        }

        sectors.clear();
        (sectors, new_sectors) = (new_sectors, sectors);
    }

    let max = sectors
        .iter()
        .max_by(|a, b| a.val.partial_cmp(&b.val).unwrap())
        .unwrap();

    Point(place_point(&max.pos, &steps, &origin))
}

fn place_point(point: &[u32], steps: &[f64], origin: &[f64]) -> Vec<f64> {
    point
        .iter()
        .zip(origin)
        .zip(steps)
        .map(|((&x, origin), step)| origin + step * x as f64)
        .collect()
}

fn refine_sector<F: Fn(&[u32]) -> f64>(
    sector: &Sector,
    steps: &[f64],
    samples: u32,
    sectors: &mut Vec<Sector>,
    acquire: F,
) {
    let mut first = true;
    let origin = sector.pos.iter().map(|&x| samples * x).collect::<Vec<_>>();
    let mut pos = vec![0; sector.pos.len()];
    loop {
        let sec_pos = origin
            .iter()
            .zip(&pos)
            .map(|(origin, x)| origin + x)
            .collect::<Vec<_>>();
        let val = if first {
            first = false;
            sector.val
        } else {
            acquire(&sec_pos)
        };
        sectors.push(Sector { pos: sec_pos, val });

        let mut carry = true;
        for digit in pos.iter_mut().rev() {
            if carry {
                carry = false;
                *digit += 1;
                if *digit == samples {
                    *digit = 0;
                    carry = true
                }
            } else {
                break;
            }
        }

        // This means that we overflowed
        if carry {
            break;
        }
    }
}
