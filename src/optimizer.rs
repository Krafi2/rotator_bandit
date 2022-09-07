#[derive(Debug)]
struct Sector {
    pos: Vec<u32>,
    val: f64,
}

pub fn grid_search<F>(
    domain: &[std::ops::Range<f64>],
    initial_samples: u32,
    samples: u32,
    retain: f64,
    refine: u32,
    mut func: F,
) -> Vec<f64>
where
    F: FnMut(&Vec<f64>) -> f64,
{
    assert!(retain <= 1., "Retain must be in the interval <0,1>");
    let d = domain.len();
    let origin = domain.iter().map(|range| range.start).collect::<Vec<_>>();
    let scale = 1. / samples as f64;

    let mut steps = domain
        .iter()
        .map(|range| (range.end - range.start) / initial_samples as f64)
        .collect::<Vec<_>>();

    let mut sectors = Vec::new();
    let mut new_sectors = Vec::new();

    // Populate the queue with initial sectors
    refine_sector(
        &Sector {
            pos: vec![0; d],
            val: func(&origin),
        },
        initial_samples,
        &mut sectors,
        |pos| {
            let point = place_point(pos, &steps, &origin);
            func(&point)
        },
    );
    for s in &mut steps {
        *s *= scale;
    }

    for _ in 0..refine {
        sectors.sort_unstable_by(|a, b| {
            a.val
                .partial_cmp(&b.val)
                .expect("Sample value was NaN!")
                .reverse()
        });
        let retain = (retain * sectors.len() as f64) as usize;
        for sector in &sectors[..retain] {
            refine_sector(sector, samples, &mut new_sectors, |pos| {
                let point = place_point(pos, &steps, &origin);
                func(&point)
            })
        }

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

    place_point(&max.pos, &steps, &origin)
}

fn place_point(point: &[u32], steps: &[f64], origin: &[f64]) -> Vec<f64> {
    point
        .iter()
        .zip(origin)
        .zip(steps)
        .map(|((&x, origin), step)| origin + step * x as f64)
        .collect()
}

fn refine_sector<F>(sector: &Sector, samples: u32, sectors: &mut Vec<Sector>, mut acquire: F)
where
    F: FnMut(&[u32]) -> f64,
{
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
