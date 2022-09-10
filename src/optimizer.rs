pub struct Search {
    pub alpha: f32,
    pub beta: f32,
    pub image: image::GrayImage,
}

pub fn grid_search<F: FnMut(f32, f32) -> f32>(
    origin: &[f32; 2],
    step: f32,
    width: usize,
    height: usize,
    mut func: F,
) -> Search {
    let mut max = (0., 0., 0.);
    let mut min = f32::MAX;
    let mut buffer = vec![0.; width * height];
    for y in 0..height {
        let beta = origin[1] + y as f32 * step;
        for x in 0..width {
            let alpha = origin[0] + x as f32 * step;
            let val = func(alpha, beta);
            if val > max.0 {
                max = (val, alpha, beta);
            }
            if val < min {
                min = val;
            }
            let i = y * width + x;
            buffer[i] = val;
        }
    }
    let scale = (max.0 - min).recip();
    let buffer = buffer
        .into_iter()
        .map(|v| ((v - min) * scale * 255.) as u8)
        .collect();

    let image = image::GrayImage::from_vec(width as u32, height as u32, buffer)
        .expect("Failed to create image");

    Search {
        alpha: max.1,
        beta: max.2,
        image,
    }
}
