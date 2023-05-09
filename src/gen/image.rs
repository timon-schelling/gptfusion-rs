use std::{path::PathBuf, sync::mpsc::Sender, time};

use anyhow::Result;
use mini_moka::unsync::Cache;
use serde::{Deserialize, Serialize};

use diffusers::pipelines::stable_diffusion;
use diffusers::transformers::clip;
use tch::{nn::Module, Device, Kind, Tensor};

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub steps: usize,
    pub acceleration_config: AccelerationConfig,
    pub version: StableDiffusionVersion,
    pub vocab_file: String,
    pub clip_weights: String,
    pub vae_weights: String,
    pub unet_weights: String,
    pub sliced_attention_size: Option<i64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            steps: 30,
            acceleration_config: AccelerationConfig::default(),
            version: StableDiffusionVersion::V1_5,
            vocab_file: "data/weights/image/stable-diffusion/v1-5/vocab.txt".to_string(),
            clip_weights: "data/weights/image/stable-diffusion/v1-5/clip.ot".to_string(),
            vae_weights: "data/weights/image/stable-diffusion/v1-5/vae.ot".to_string(),
            unet_weights: "data/weights/image/stable-diffusion/v1-5/unet.ot".to_string(),
            sliced_attention_size: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum StableDiffusionVersion {
    V1_5,
    V2_1,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Args {
    pub prompt: String,
    pub seed: i64,
    pub path: PathBuf,
}

fn build_stable_diffusion_config(
    version: StableDiffusionVersion,
    sliced_attention_size: Option<i64>,
    width: Option<i64>,
    height: Option<i64>,
) -> stable_diffusion::StableDiffusionConfig {
    match version {
        StableDiffusionVersion::V1_5 => {
            stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::V2_1 => {
            stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Workload {
    Clip,
    Vae,
    Unet,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AccelerationConfig {
    clip: bool,
    vae: bool,
    unet: bool,
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            clip: true,
            vae: true,
            unet: true,
        }
    }
}

impl AccelerationConfig {
    fn build_device_for(&self, workload: Workload) -> Device {
        let is_accelerated = match workload {
            Workload::Clip => self.clip,
            Workload::Vae => self.vae,
            Workload::Unet => self.unet,
        };
        if is_accelerated {
            tch::Device::cuda_if_available()
        } else {
            tch::Device::Cpu
        }
    }
}

pub fn batch_default(args: impl IntoIterator<Item = Args>, status: Sender<Info>) -> Result<()> {
    batch(args, Config::default(), status)
}

pub fn batch(
    args: impl IntoIterator<Item = Args>,
    config: Config,
    status: Sender<Info>,
) -> Result<()> {
    tch::autocast(true, || exec_batch(args, config, status))
}

#[derive(Debug, Clone)]
pub enum Info {
    System {
        cuda: bool,
        cudnn: bool,
        mps: bool,
    },
    Building(Workload),
    TimestepStart(usize),
    TimestepDone(time::Duration),
    ImageStart(Args),
    ImageDone(time::Duration),
    Done,
}

fn exec_batch(
    args: impl IntoIterator<Item = Args>,
    config: Config,
    info: Sender<Info>,
) -> Result<()> {
    let Config {
        width,
        height,
        steps,
        acceleration_config,
        version,
        vocab_file,
        clip_weights,
        vae_weights,
        unet_weights,
        sliced_attention_size,
    } = config;

    tch::maybe_init_cuda();

    info.send(Info::System {
        cuda: tch::Cuda::is_available(),
        cudnn: tch::Cuda::cudnn_is_available(),
        mps: tch::utils::has_mps(),
    })?;

    let sd_config = build_stable_diffusion_config(version, sliced_attention_size, width, height);

    let clip_device = acceleration_config.build_device_for(Workload::Clip);
    let vae_device = acceleration_config.build_device_for(Workload::Vae);
    let unet_device = acceleration_config.build_device_for(Workload::Unet);
    let scheduler = sd_config.build_scheduler(steps);

    info.send(Info::Building(Workload::Clip))?;
    let text_model = sd_config.build_clip_transformer(&clip_weights, clip_device)?;

    info.send(Info::Building(Workload::Vae))?;
    let vae = sd_config.build_vae(&vae_weights, vae_device)?;

    info.send(Info::Building(Workload::Unet))?;
    let unet = sd_config.build_unet(&unet_weights, unet_device, 4)?;

    let tokenizer = clip::Tokenizer::create(vocab_file, &sd_config.clip)?;

    let tockenize = |text: &str| -> Result<Tensor> {
        let tokens = tokenizer.encode(&text)?;
        let tokens: Vec<i64> = tokens.into_iter().map(|x| x as i64).collect();
        Ok(Tensor::of_slice(&tokens).view((1, -1)).to(clip_device))
    };

    let uncond_tokens = tockenize("")?;

    let mut text_embeddings_cache: Cache<String, Tensor> = Cache::new(100);

    let mut text_embeddings = |text: String| -> Result<Tensor> {
        if let Some(cache_hit) = text_embeddings_cache.get(&text) {
            return Ok(cache_hit.shallow_clone());
        };
        let tokens = tockenize(&text)?;
        let text_embeddings = text_model.forward(&tokens);
        let uncond_embeddings = text_model.forward(&uncond_tokens);
        Ok(Tensor::cat(&[uncond_embeddings, text_embeddings], 0).to(unet_device))
    };

    for args in args {

        info.send(Info::ImageStart(args.clone()))?;

        let image_start = time::Instant::now();

        let Args { prompt, seed, path } = args;
        let text_embeddings = text_embeddings(prompt)?;

        let _no_grad_guard = tch::no_grad_guard();

        tch::manual_seed(seed);
        let mut latents = Tensor::randn(
            &[1, 4, sd_config.height / 8, sd_config.width / 8],
            (Kind::Float, unet_device),
        );

        // scale the initial noise by the standard deviation required by the scheduler
        latents *= scheduler.init_noise_sigma();

        for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {

            info.send(Info::TimestepStart(timestep_index))?;

            let timestep_start = time::Instant::now();

            let latent_model_input = Tensor::cat(&[&latents, &latents], 0);

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep);
            let noise_pred = unet.forward(&latent_model_input, timestep as f64, &text_embeddings);
            let noise_pred = noise_pred.chunk(2, 0);
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            const GUIDANCE_SCALE: f64 = 7.5;
            let noise_pred =
                noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE;
            latents = scheduler.step(&noise_pred, timestep, &latents);

            let elapsed = timestep_start.elapsed();
            info.send(Info::TimestepDone(elapsed))?;
        }


        let latents = latents.to(vae_device);
        let image = vae.decode(&(&latents / 0.18215));
        let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
        let image = (image * 255.).to_kind(Kind::Uint8);
        tch::vision::image::save(&image, path)?;

        let elapsed = image_start.elapsed();

        info.send(Info::ImageDone(elapsed))?;
    }

    info.send(Info::Done)?;

    Ok(())
}