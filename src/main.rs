use std::{
    fs,
    io::{stdout, Write},
    path::PathBuf,
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result};

use axum::{extract::State, routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};
use sliding_window_alt::SlidingWindow;

use gen::image::*;
use uuid::Uuid;

mod gen;

const IMAGES_PATH: &str = "out/images";

#[derive(Clone)]
struct ApiState {
    image_gen_channel: Arc<Mutex<Sender<Args>>>,
}

fn api() -> Router<ApiState> {
    Router::new().nest("/api", Router::new().route("/generate", post(generate)))
}

#[derive(Serialize, Deserialize)]
struct GenerateRequest {
    seed: Option<i64>,
    prompt: String,
}

#[derive(Serialize, Deserialize)]
struct GenerateResponse {
    uuid: Uuid,
}

#[derive(Serialize, Deserialize)]
struct ImageMetadata {
    prompt: String,
    seed: i64,
}

async fn generate(
    State(state): State<ApiState>,
    Json(payload): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let image_gen_channel_guard = state.image_gen_channel.lock().unwrap();
    let image_gen_channel = image_gen_channel_guard.clone();
    drop(image_gen_channel_guard);

    let seed = payload.seed.unwrap_or_else(|| rand::random());
    let prompt = payload.prompt;

    let uuid = Uuid::new_v4();

    let folder = PathBuf::from(IMAGES_PATH);
    let path = folder.join(uuid.to_string());
    let metadata_path = path.with_extension("json");
    let image_path = path.with_extension("png");

    let metadata = ImageMetadata {
        prompt: prompt.clone(),
        seed,
    };

    fs::create_dir_all(folder).unwrap();

    fs::write(metadata_path, serde_json::to_string(&metadata).unwrap()).unwrap();

    let args = Args {
        seed,
        prompt,
        path: image_path,
    };

    image_gen_channel.send(args).unwrap();

    Json(GenerateResponse { uuid })
}

fn main() {
    let (status_tx, status_rx) = mpsc::channel::<Status>();
    let (args_tx, args_rx) = mpsc::channel::<Args>();

    let print_info = spawn_print_info_thread(status_rx);

    let rest_api = spawn_rest_api_thread(args_tx);

    let image_gen = spawn_image_gen_thread(args_rx, status_tx);

    match image_gen.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("image gen thread failed: {}", e);
        }
        Err(e) => {
            eprintln!("image gen thread panicked: {:?}", e);
        }
    };
    match rest_api.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("rest api thread failed: {}", e);
        }
        Err(e) => {
            eprintln!("rest api thread panicked: {:?}", e);
        }
    };
    match print_info.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("print info thread failed: {}", e);
        }
        Err(e) => {
            eprintln!("print info thread panicked: {:?}", e);
        }
    };
}

async fn run_api_server(args_tx: Sender<Args>) -> Result<()> {
    let state = ApiState {
        image_gen_channel: Arc::new(Mutex::new(args_tx)),
    };

    axum::Server::bind(&"0.0.0.0:4200".parse().unwrap())
        .serve(api().with_state(state).into_make_service())
        .await
        .context("failed to run api server")
}

fn spawn_rest_api_thread(args_tx: Sender<Args>) -> thread::JoinHandle<Result<()>> {
    thread::spawn(move || -> Result<()> {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(run_api_server(args_tx))?;
        Ok(())
    })
}

fn spawn_image_gen_thread(
    args_rx: Receiver<Args>,
    status_tx: Sender<Status>,
) -> thread::JoinHandle<Result<()>> {
    let image_gen = thread::spawn(move || -> Result<()> {
        batch_default(args_rx, status_tx)?;
        Ok(())
    });
    image_gen
}

fn spawn_print_info_thread(info_rx: Receiver<Status>) -> thread::JoinHandle<Result<()>> {
    thread::spawn(move || -> Result<()> {
        print_info(info_rx)?;
        Ok(())
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
                    } => {}
                    Status::Building(_) => {}
                    Status::TimestepStart(n) => {
                        step = n + 1;
                    }
                    Status::TimestepDone(d) => {
                        timestep_durations.push(Some(d));
                    }
                    Status::ImageStart(m) => {
                        metadata = Some(m);
                    }
                    Status::ImageDone(d) => {
                        image_durations.push(Some(d));
                        step = 0;
                    }
                    Status::Done => break,
                }
                if let Status::Done = info {
                    break;
                }
            }
            Err(_) => break,
        }

        // print!("{esc}[2J{esc}[1;1H", esc = 27 as char);

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
            print!(
                "Seed: {:x} Prompt: {: <50}\n",
                metadata.seed, metadata.prompt
            );
        }
        stdout.flush().unwrap();
    }

    Ok(())
}
