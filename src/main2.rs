use anyhow::Result;

use std::time;

use gen::image::*;
use rand::Rng;

mod gen;

fn main2() {
    gen::text::test().unwrap();
}

fn main() -> Result<()> {

    let start_time = chrono::offset::Utc::now();

    let promts = vec![
        // "photo of the early universe",
        "photo of the early morning hours",
        // "photo of the earth from space",
        "photo of mountains",
        "surreal landscape",
        // "futuristic technology",
        // "image of a mythical creature that has never been seen before",
        "image of an alien planet with unique flora and fauna",
        // "cityscape that merges different architectural styles",
        // "futuristic vehicle",
    ];

    let mut rng = rand::thread_rng();

    let iter = gen::image::batch_default((0..1000).map(|_| {

        let seed = rng.gen::<i64>();

        promts.iter().map(move |prompt| {
            let folder = std::path::PathBuf::from(format!("out/[{:?}]/({})", start_time, prompt));
            std::fs::create_dir_all(&folder).unwrap();

            let path = folder.join(format!("({:x}).png", seed));

            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }

            Args { prompt: prompt.to_string(), seed, path }
        })

    }).flatten().collect::<Vec<_>>())?;

    let mut image_time = time::Instant::now();

    let mut timestep_time = time::Instant::now();

    for step in iter {
        let step = step?;
        match step {
            Step::Tokenize => {
                todo!()
            },
            Step::Timestep => {
                println!("Timestep: {:?}ms", timestep_time.elapsed().as_millis());
                timestep_time = time::Instant::now();
            },
            Step::Image => {
                println!("Image: {:?}ms", image_time.elapsed().as_millis());
                image_time = time::Instant::now();
            },
        }
    }

    Ok(())
}
