const SAMPLES: usize = 100;
const PROMTS: &'static [&'static str] = &[
    "photo of cat, unreal engine",
    "photo of mountains",
    "photo of a city at night, cyberpunk",
    "Cute small cat sitting in a movie theater eating chicken wiggs watching a movie, unreal engine, cozy indoor lighting, artstation, detailed, digital painting, cinematic,character design by mark ryden and pixar and hayao miyazaki, unreal 5, daz, hyperrealistic, octane render"
];

use std::{
    fs,
    io::{stdout, Write},
    path::PathBuf,
    sync::mpsc::{self, Receiver},
    thread,
    time::Duration,
};

use anyhow::Result;

use sliding_window_alt::SlidingWindow;

use gen::image::*;
use rand::Rng;

mod gen;

fn main() {
    let start_time = chrono::offset::Utc::now();

    let (status_tx, status_rx) = mpsc::channel::<Status>();
    let (args_tx, args_rx) = mpsc::channel::<Args>();

    let print_info = spawn_print_info_thread(status_rx);

    let args_gen = thread::spawn(move || -> Result<()> {
        let mut rng = rand::thread_rng();

        for _ in 0..SAMPLES {
            let seed = rng.gen::<i64>();

            for (i, prompt) in PROMTS.iter().enumerate() {
                let shortend_prompt = prompt.to_string().chars().take(64).collect::<String>();

                let folder = PathBuf::from(format!(
                    "out/[{:?}]/{}-({})",
                    start_time,
                    i + 1,
                    shortend_prompt
                ));
                fs::create_dir_all(&folder)?;

                let path = folder.join(format!("({:x}).png", seed));

                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }

                args_tx.send(Args {
                    prompt: prompt.to_string(),
                    seed,
                    path,
                })?;
            }
        }

        Ok(())
    });

    let image_gen = thread::spawn(move || -> Result<()> {
        batch_default(args_rx, status_tx)?;
        Ok(())
    });

    image_gen.join().unwrap().unwrap();
    args_gen.join().unwrap().unwrap();
    print_info.join().unwrap();
}

fn spawn_print_info_thread(info_rx: Receiver<Status>) -> thread::JoinHandle<Result<()>> {
    thread::spawn(move || -> Result<()> {
        print_info(info_rx)
    })
}

fn print_info(info_rx: Receiver<Status>) -> Result<()> {
    let mut step: usize = 0;

    const TIMESTEPS_PER_IMAGE: usize = 30;
    const IMAGE_DURATIONS_TO_KEEP: usize = 10;
    const TIMESTEP_DURATIONS_TO_KEEP: usize = IMAGE_DURATIONS_TO_KEEP * TIMESTEPS_PER_IMAGE;

    let mut image_durations: SlidingWindow<Option<Duration>> =
        SlidingWindow::new(IMAGE_DURATIONS_TO_KEEP, None);
    let mut timestep_durations: SlidingWindow<Option<Duration>> =
        SlidingWindow::new(TIMESTEP_DURATIONS_TO_KEEP, None);

    let mut metadata: Option<Metadata> = None;

    let mut stdout = stdout();

    loop {
        match info_rx.recv() {
            Ok(info) => {
                match info.clone() {
                    Status::System {
                        cuda: _,
                        cudnn: _,
                        mps: _,
                    } => {},
                    Status::Building(_) => {},
                    Status::TimestepStart(n) => {
                        step = n + 1;
                    },
                    Status::TimestepDone(d) => {
                        timestep_durations.push(Some(d));
                    },
                    Status::ImageStart(m) => {
                        metadata = Some(m);
                    },
                    Status::ImageDone(d) => {
                        image_durations.push(Some(d));
                        step = 0;

                        if let Some(mut metadata) = metadata.clone() {

                            let mut path = metadata.out.clone();
                            path.set_extension("json");

                            let file_name = metadata.out.file_name().unwrap();
                            let file_name = file_name.to_str().unwrap();

                            metadata.out = file_name.into();

                            let mut contents = serde_json::to_string_pretty(&metadata).unwrap();
                            contents.push('\n');

                            fs::write(path, contents).unwrap();
                        }
                    },
                    Status::Done => break,
                }
                if let Status::Done = info {
                    break;
                }
            }
            Err(_) => break,
        }

        print!("{esc}[2J{esc}[1;1H", esc = 27 as char);

        let image_durations: Vec<Duration> = image_durations
            .iter()
            .filter_map(|e| e.as_ref())
            .cloned()
            .collect();
        let average_image_duration = if image_durations.len() > 0 {
            image_durations.iter().sum::<Duration>() / image_durations.len() as u32
        } else {
            Duration::from_secs(0)
        };

        let timestep_durations: Vec<Duration> = timestep_durations
            .iter()
            .filter_map(|e| e.as_ref())
            .cloned()
            .collect();
        let average_timestep_duration = if timestep_durations.len() > 0 {
            timestep_durations.iter().sum::<Duration>() / timestep_durations.len() as u32
        } else {
            Duration::from_secs(0)
        };

        print!(
            "Image time: {: >4}ms Step time: {: >3}ms Step: {: >2}\n",
            average_image_duration.as_millis(),
            average_timestep_duration.as_millis(),
            step,
        );
        if let Some(metadata) = &metadata {
            print!("Seed: {:x} Prompt: {: <50}\n", metadata.seed, metadata.prompt);
        }
        stdout.flush().unwrap();
    }

    Ok(())
}
