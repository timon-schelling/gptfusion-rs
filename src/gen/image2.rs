use std::rc::Rc;
use std::{path::PathBuf, time};

use anyhow::Result;
use diffusers::models::unet_2d::UNet2DConditionModel;
use diffusers::models::vae::AutoEncoderKL;
use diffusers::schedulers::ddim::DDIMScheduler;
use mini_moka::unsync::Cache;
use serde::{Deserialize, Serialize};

use diffusers::transformers::clip;
use diffusers::{pipelines::stable_diffusion, transformers::clip::ClipTextTransformer};
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

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
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

#[derive(Serialize, Deserialize, Debug)]
enum Workload {
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

pub fn batch_default(args: impl IntoIterator<Item = Args>) -> Result<()> {
    batch(args, Config::default())
}

pub fn batch(args: impl IntoIterator<Item = Args>, config: Config) -> Result<()> {
    tch::autocast(true, || exec_batch(args, config))
}

pub struct BatchExecutor<'a> {
    config: Config,
    args_iter: Box<dyn Iterator<Item = Args> + 'a>,
    args: Option<Rc<Args>>,
    tokenizer: Tokenizer,
    generator_config: GeneratorConfig,
    generator: Option<Rc<Generator>>,
    renderer: Renderer,
}

enum BatchExecutorStep {
    Timestep,
    Image,
    Done,
}

impl<'a> BatchExecutor<'a> {
    pub fn new(
        args: impl Iterator<Item = Args> + 'a,
        config: Config,
    ) -> Result<Self> {
        tch::maybe_init_cuda();

        println!("Cuda available: {}", tch::Cuda::is_available());
        println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
        println!("MPS available: {}", tch::utils::has_mps());

        let sd_config = build_stable_diffusion_config(
            config.version,
            config.sliced_attention_size,
            config.width,
            config.height,
        );

        let clip_device = config.acceleration_config.build_device_for(Workload::Clip);
        let vae_device = config.acceleration_config.build_device_for(Workload::Vae);
        let unet_device = config.acceleration_config.build_device_for(Workload::Unet);

        let scheduler = sd_config.build_scheduler(config.steps);

        let clip_model = sd_config.build_clip_transformer(&config.clip_weights, clip_device)?;
        let vae_model = sd_config.build_vae(&config.vae_weights, vae_device)?;
        let unet_model = sd_config.build_unet(&config.unet_weights, unet_device, 4)?;

        let clip_tokenizer = clip::Tokenizer::create(&config.vocab_file, &sd_config.clip)?;
        let tokenizer = Tokenizer::new(clip_tokenizer, clip_model, clip_device)?;

        let generator_config = GeneratorConfig {
            model: unet_model,
            sd_config: sd_config,
            scheduler: scheduler,
            device: unet_device,
        };

        let renderer = Renderer {
            model: vae_model,
            device: vae_device,
        };

        Ok(Self {
            config,
            args_iter: Box::new(args),
            args: None,
            generator_config,
            generator: None,
            tokenizer,
            renderer
        })
    }
}

impl BatchExecutor<'_> {
    fn next(&mut self) -> Result<BatchExecutorStep> {

        let args = match self.args.clone() {
            Some(args) => args,
            None => {
                let args = match self.args_iter.next() {
                    Some(args) => Rc::new(args),
                    None => return Ok(BatchExecutorStep::Done),
                };
                self.args = Some(args.clone());
                args
            }
        };

        let mut generator = match self.generator.clone() {
            Some(generator) => generator,
            None => {
                let text_embeddings = self.tokenizer.encode(args.prompt.clone())?;
                let generator = Rc::new(Generator::new(text_embeddings, args.seed));
                self.generator = Some(generator.clone());
                generator
            }
        };

        let generator_step = generator.next(self.generator_config)?;

        let step = match generator_step {
            GeneratorStep::Timestep => BatchExecutorStep::Timestep,
            GeneratorStep::Image(image) => {
                let image = self.renderer.render(image, args.path)?;
                self.generator = None;
                self.args = None;
                BatchExecutorStep::Image
            }
            GeneratorStep::Done => unreachable!("Generator should not return Done")
        };

        Ok(step)
    }
}

struct Tokenizer {
    tokenizer: clip::Tokenizer,
    model: ClipTextTransformer,
    device: Device,
    uncond_embeddings: Option<Tensor>,
    cache: Cache<String, Tensor>,
}

impl Tokenizer {
    fn new(tokenizer: clip::Tokenizer, model: ClipTextTransformer, device: Device) -> Result<Self> {
        Ok(Self {
            tokenizer,
            model,
            device,
            uncond_embeddings: None,
            cache: Cache::new(100),
        })
    }

    fn encode(&mut self, text: String) -> Result<Tensor> {
        let cached = self.cache.get(&text);
        if let Some(cached) = cached {
            return Ok(cached.shallow_clone());
        }

        let tockenize = |text: &str| -> Result<Tensor> {
            let tokens = self.tokenizer.encode(&text)?;
            let tokens: Vec<i64> = tokens.into_iter().map(|x| x as i64).collect();
            Ok(Tensor::of_slice(&tokens).view((1, -1)).to(self.device))
        };

        let uncond_embeddings = match &self.uncond_embeddings {
            Some(uncond_embeddings) => uncond_embeddings.shallow_clone(),
            None => {
                let tokens = tockenize("")?;
                let embeddings = self.model.forward(&tokens);
                self.uncond_embeddings = Some(embeddings.shallow_clone());
                embeddings
            }
        };

        let tokens = tockenize(&text)?;
        let text_embeddings = self.model.forward(&tokens);
        let embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0).to(self.device);

        self.cache.insert(text, embeddings.shallow_clone());

        Ok(embeddings)
    }
}

struct GeneratorConfig {
    model: UNet2DConditionModel,
    sd_config: stable_diffusion::StableDiffusionConfig,
    scheduler: DDIMScheduler,
    device: Device,
}

enum GeneratorStep {
    Timestep,
    Image(Tensor),
    Done,
}

struct Generator {
    seed: i64,
    text_embeddings: Tensor,
    latents: Option<Tensor>,
    timesteps: Option<Box<dyn Iterator<Item = usize>>>
}

impl Generator {
    fn new(text_embeddings: Tensor, seed: i64) -> Self {
        Self {
            seed,
            text_embeddings,
            latents: None,
            timesteps: None,
        }
    }
}

impl Generator {
    fn next(&mut self, config: GeneratorConfig) -> Result<GeneratorStep> {
        let latents = match self.latents {
            Some(latents) => latents,
            None => {
                tch::manual_seed(self.seed);
                let mut latents = Tensor::randn(
                    &[1, 4, config.sd_config.height / 8, config.sd_config.width / 8],
                    (Kind::Float, config.device),
                );
                latents *= config.scheduler.init_noise_sigma();
                latents
            }
        };

        let timestep = match self.timesteps {
            Some(timesteps) => timesteps.next(),
            None => {
                let timestep = config.scheduler.timesteps().iter().map(|a| *a);
                self.timesteps = Some(Box::new(timestep));
                timestep.next()
            },
        };

        let timestep = match timestep {
            Some(timestep) => timestep,
            None => return Ok(GeneratorStep::Done),
        };

        let latent_model_input = Tensor::cat(&[&latents, &latents], 0);

        let latent_model_input = config.scheduler.scale_model_input(latent_model_input, timestep);
        let noise_pred = config.model.forward(&latent_model_input, timestep as f64, &self.text_embeddings);
        let noise_pred = noise_pred.chunk(2, 0);
        let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
        const GUIDANCE_SCALE: f64 = 7.5;
        let noise_pred =
            noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE;
        latents = config.scheduler.step(&noise_pred, timestep, &latents);

        todo!()
    }
}

struct Renderer {
    model: AutoEncoderKL,
    device: Device,
}

impl Renderer {
    fn render(&self, image: Tensor, path: PathBuf) -> Result<Tensor> {
        let latents = image.to(self.device);
        let image = self.model.decode(&(&latents / 0.18215));
        let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
        let image = (image * 255.).to_kind(Kind::Uint8);
        tch::vision::image::save(&image, path)?;
        Ok(image)
    }
}

fn exec_batch(args: impl IntoIterator<Item = Args>, config: Config) -> Result<()> {

    tch::maybe_init_cuda();

    let mut batch_executor = BatchExecutor::new(args.into_iter(), config)?;

    let mut image_time = time::Instant::now();

    let mut timestep_time = time::Instant::now();

    loop {
        match batch_executor.next()? {
            BatchExecutorStep::Timestep => {
                println!("Timestep: {:?}ms", timestep_time.elapsed().as_millis());
                timestep_time = time::Instant::now();
            },
            BatchExecutorStep::Image => {
                println!("Image: {:?}ms", image_time.elapsed().as_millis());
                image_time = time::Instant::now();
            },
            BatchExecutorStep::Done => break,
        }
    }

    Ok(())
}
