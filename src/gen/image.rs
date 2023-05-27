use std::{
    path::PathBuf,
    sync::mpsc::{Receiver, Sender},
    time,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tch::Tensor;

pub(crate) mod backend;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Config {
    pub width: i64,
    pub height: i64,
    pub steps: usize,
    pub version: Version,
    pub vocab_file: PathBuf,
    pub clip_weights: PathBuf,
    pub vae_weights: PathBuf,
    pub unet_weights: PathBuf,
    pub sliced_attention_size: Option<i64>,
}

impl Config {
    fn from_version(version: Version) -> Config {
        let mut base_path = PathBuf::from("data/weights/image/stable-diffusion");
        base_path.push(match version {
            Version::V1_5 => "v1-5",
            Version::V2_1 => "v2-1",
        });

        Self {
            width: 512,
            height: 512,
            steps: 30,
            version,
            vocab_file: base_path.join("vocab.txt"),
            clip_weights: base_path.join("clip.safetensors").into(),
            vae_weights: base_path.join("vae.safetensors").into(),
            unet_weights: base_path.join("unet.safetensors").into(),
            sliced_attention_size: None,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::from_version(Version::V2_1)
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum Version {
    V1_5,
    V2_1,
}

#[derive(Clone, Debug)]
pub struct BackendConfigBuilder {
    pub clip_config: backend::transformers::clip::Config,
    pub vae_config: backend::models::vae::AutoEncoderKLConfig,
    pub unet_config: backend::models::unet_2d::UNet2DConditionModelConfig,
    pub scheduler_config: backend::schedulers::ddim::DDIMSchedulerConfig,
}

impl BackendConfigBuilder {
    pub fn new(version: Version, sliced_attention_size: Option<i64>) -> Self {
        use backend::models::{unet_2d, vae};
        use backend::transformers::clip;

        let clip_config = match version {
            Version::V1_5 => clip::Config::v1_5(),
            Version::V2_1 => clip::Config::v2_1(),
        };

        let vae_config = vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
        };

        let unet_config = {
            let blocks = match version {
                Version::V1_5 => [
                    (320, true, 8),
                    (640, true, 8),
                    (1280, true, 8),
                    (1280, false, 8),
                ],
                Version::V2_1 => [
                    (320, true, 5),
                    (640, true, 10),
                    (1280, true, 20),
                    (1280, false, 20),
                ],
            }
            .iter()
            .map(
                |(out_channels, use_cross_attn, attention_head_dim)| unet_2d::BlockConfig {
                    out_channels: *out_channels,
                    use_cross_attn: *use_cross_attn,
                    attention_head_dim: *attention_head_dim,
                },
            )
            .collect::<Vec<unet_2d::BlockConfig>>();

            let cross_attention_dim = match version {
                Version::V1_5 => 768,
                Version::V2_1 => 1024,
            };

            let use_linear_projection = match version {
                Version::V1_5 => false,
                Version::V2_1 => true,
            };

            unet_2d::UNet2DConditionModelConfig {
                blocks: blocks,
                center_input_sample: false,
                cross_attention_dim,
                downsample_padding: 1,
                flip_sin_to_cos: true,
                freq_shift: 0.,
                layers_per_block: 2,
                mid_block_scale_factor: 1.,
                norm_eps: 1e-5,
                norm_num_groups: 32,
                sliced_attention_size,
                use_linear_projection,
            }
        };

        let scheduler_config = backend::schedulers::ddim::DDIMSchedulerConfig {
            prediction_type: backend::schedulers::PredictionType::VPrediction,
            ..Default::default()
        };

        BackendConfigBuilder {
            clip_config,
            vae_config,
            unet_config,
            scheduler_config,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Args {
    pub prompt: String,
    pub seed: i64,
    pub path: PathBuf,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum Workload {
    Clip,
    Vae,
    Unet,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Metadata {
    pub prompt: String,
    pub seed: i64,
    pub width: i64,
    pub height: i64,
    pub steps: usize,
    pub out: PathBuf,
}

#[derive(Clone, Debug)]
pub enum Status {
    System { cuda: bool, cudnn: bool, mps: bool },
    Building(Workload),
    TimestepStart(usize),
    TimestepDone(time::Duration),
    ImageStart(Metadata),
    ImageDone(time::Duration),
    Done,
}

pub fn batch_default(args: Receiver<Args>, status: Sender<Status>) -> Result<()> {
    batch(Default::default(), args, status)
}

pub fn batch(config: Config, args: Receiver<Args>, status: Sender<Status>) -> Result<()> {
    tch::autocast(true, || exec_batch(args, config, status))
}

fn exec_batch(args: Receiver<Args>, config: Config, status: Sender<Status>) -> Result<()> {
    let Config {
        width,
        height,
        steps,
        version,
        vocab_file,
        clip_weights,
        vae_weights,
        unet_weights,
        sliced_attention_size,
    } = config.clone();

    tch::maybe_init_cuda();
    let _no_grad_guard = tch::no_grad_guard();

    status.send(Status::System {
        cuda: tch::Cuda::is_available(),
        cudnn: tch::Cuda::cudnn_is_available(),
        mps: tch::utils::has_mps(),
    })?;

    let clip_device = tch::Device::cuda_if_available();
    let vae_device = tch::Device::cuda_if_available();
    let unet_device = tch::Device::cuda_if_available();

    let BackendConfigBuilder {
        clip_config,
        vae_config,
        unet_config,
        scheduler_config,
    } = BackendConfigBuilder::new(version, sliced_attention_size);

    let clip =
        clip::Clip::<clip::ClipStateNew>::new(clip_config, clip_device, vocab_file, clip_weights);
    let vae = vae::Vae::<vae::VaeStateNew>::new(vae_config, vae_device, vae_weights);
    let unet = unet::Unet::<unet::UnetStateNew>::new(
        width,
        height,
        steps,
        unet_config,
        scheduler_config,
        unet_device,
        unet_weights,
    );

    status.send(Status::Building(Workload::Clip))?;
    let mut clip = clip.build()?;

    status.send(Status::Building(Workload::Vae))?;
    let vae = vae.build()?;

    status.send(Status::Building(Workload::Unet))?;
    let unet = unet.build()?;

    let mut arg: Option<Args> = None;
    let mut embeddings: Option<Tensor> = None;

    enum UnetWithState {
        Build(unet::Unet<unet::UnetStateBuild>),
        Image(unet::Unet<unet::UnetStateImage>),
        Done(unet::Unet<unet::UnetStateImageDone>),
    }

    let mut unet = UnetWithState::Build(unet);

    let mut image: Option<Tensor> = None;

    let mut image_start = time::Instant::now();

    let mut timestep = 0;

    loop {
        let Args { prompt, seed, path } = match arg.clone() {
            Some(arg) => arg,
            None => match args.recv() {
                Ok(a) => {
                    arg = Some(a);
                    continue;
                }
                Err(_) => break,
            },
        };

        let embeds = match embeddings {
            Some(ref e) => e,
            None => {
                let e = clip.embed(&prompt)?;
                embeddings = Some(e);
                continue;
            }
        };

        unet = match unet {
            UnetWithState::Build(unet) => {
                status.send(Status::ImageStart(Metadata {
                    prompt: prompt.clone(),
                    seed,
                    width,
                    height,
                    steps,
                    out: path.clone(),
                }))?;

                image_start = time::Instant::now();

                UnetWithState::Image(unet.image(embeds, seed))
            }
            UnetWithState::Image(mut unet) => {

                let timestep_start = time::Instant::now();

                status.send(Status::TimestepStart(timestep))?;

                let unet = match unet.step() {
                    Some(()) => UnetWithState::Done(unet.finish()),
                    None => UnetWithState::Image(unet),
                };

                status.send(Status::TimestepDone(timestep_start.elapsed()))?;

                timestep += 1;

                if let UnetWithState::Done(_) = unet {
                    status.send(Status::ImageDone(image_start.elapsed()))?;
                }

                unet
            },
            UnetWithState::Done(unet) => {
                arg = None;
                embeddings = None;
                image = Some(unet.image());
                let unet = unet.reset()?;
                UnetWithState::Build(unet)
            }
        };

        if let Some(img) = image {
            let img = vae.decode(img)?;

            tch::vision::image::save(&img, path)?;
            image = None;

            status.send(Status::ImageDone(image_start.elapsed()))?;

            timestep = 0;

            image_start = time::Instant::now();
        }
    }

    status.send(Status::Done)?;

    Ok(())
}

mod clip {
    use std::{marker::PhantomData, path::PathBuf};

    use anyhow::Result;

    use super::backend::transformers::clip::{self, Tokenizer};
    use tch::{nn::Module, Device, Tensor};

    pub(crate) trait ClipState {}

    pub(crate) struct ClipStateNew;
    impl ClipState for ClipStateNew {}
    pub(crate) struct ClipStateBuild;
    impl ClipState for ClipStateBuild {}

    pub(crate) struct Clip<T>
    where
        T: ClipState,
    {
        config: clip::Config,
        device: Device,
        vocab: PathBuf,
        weights: PathBuf,
        model: Option<clip::ClipTextTransformer>,
        tokenizer: Option<clip::Tokenizer>,
        uncond_embeddings: Option<Tensor>,
        _state: PhantomData<T>,
    }

    impl<T: ClipState> Clip<T> {
        pub(crate) fn new(
            config: clip::Config,
            device: Device,
            vocab: PathBuf,
            weights: PathBuf,
        ) -> Clip<ClipStateNew> {
            Clip {
                config,
                device,
                vocab,
                weights,
                model: None,
                tokenizer: None,
                uncond_embeddings: None,
                _state: PhantomData,
            }
        }
    }

    impl Clip<ClipStateNew> {
        pub(crate) fn build(self) -> Result<Clip<ClipStateBuild>> {
            let tokenizer = clip::Tokenizer::create(self.vocab.clone(), &self.config)?;

            let mut vs = tch::nn::VarStore::new(self.device);
            let model = clip::ClipTextTransformer::new(vs.root(), &self.config);
            vs.load(self.weights.clone())?;

            let uncond_tokens = tokenize(&"".to_string(), &tokenizer)?.to(self.device);
            let uncond_embeddings = model.forward(&uncond_tokens);

            Ok(Clip {
                config: self.config,
                device: self.device,
                vocab: self.vocab,
                weights: self.weights,
                model: Some(model),
                tokenizer: Some(tokenizer),
                uncond_embeddings: Some(uncond_embeddings),
                _state: PhantomData,
            })
        }
    }

    impl Clip<ClipStateBuild> {
        pub(crate) fn embed(&mut self, text: &String) -> Result<Tensor> {
            let model = self.model.as_ref().expect("model not initialized");
            let tokenizer = self.tokenizer.as_ref().expect("tokenizer not initialized");
            let uncond_embeddings = self
                .uncond_embeddings
                .as_ref()
                .expect("uncond_embeddings not initialized");

            let tokens = tokenize(text, tokenizer)?.to(self.device);
            let embeddings = model.forward(&tokens);

            let embeddings = Tensor::cat(&[uncond_embeddings, &embeddings], 0).to(self.device);

            Ok(embeddings)
        }
    }

    fn tokenize(text: &String, tokenizer: &Tokenizer) -> Result<Tensor> {
        let tokens = tokenizer.encode(&text)?;
        let tokens = tokens.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
        let tokens = Tensor::from_slice(&tokens).view((1, -1));
        Ok(tokens)
    }
}

mod vae {
    use std::{marker::PhantomData, path::PathBuf};

    use anyhow::Result;

    use super::backend::models::vae;
    use tch::{nn, Device, Kind, Tensor};

    pub(crate) trait VaeState {}

    pub(crate) struct VaeStateNew;
    impl VaeState for VaeStateNew {}
    pub(crate) struct VaeStateBuild;
    impl VaeState for VaeStateBuild {}

    pub(crate) struct Vae<T>
    where
        T: VaeState,
    {
        config: vae::AutoEncoderKLConfig,
        device: Device,
        weights: PathBuf,
        model: Option<vae::AutoEncoderKL>,
        _state: PhantomData<T>,
    }

    impl<T: VaeState> Vae<T> {
        pub(crate) fn new(
            config: vae::AutoEncoderKLConfig,
            device: Device,
            weights: PathBuf,
        ) -> Vae<VaeStateNew> {
            Vae {
                config,
                device,
                weights,
                model: None,
                _state: PhantomData,
            }
        }
    }

    impl Vae<VaeStateNew> {
        pub(crate) fn build(self) -> Result<Vae<VaeStateBuild>> {
            let mut vs = nn::VarStore::new(self.device);
            let model = vae::AutoEncoderKL::new(vs.root(), 3, 3, self.config.clone());
            vs.load(self.weights.clone())?;

            Ok(Vae {
                config: self.config,
                device: self.device,
                weights: self.weights,
                model: Some(model),
                _state: PhantomData,
            })
        }
    }

    impl Vae<VaeStateBuild> {
        pub(crate) fn decode(&self, image: Tensor) -> Result<Tensor> {
            let model = self.model.as_ref().expect("model not initialized");

            let image = image.to(self.device);
            let image = model.decode(&(&image / 0.18215));
            let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
            let image = (image * 255.).to_kind(Kind::Uint8);

            Ok(image)
        }
    }
}

mod unet {
    use std::{marker::PhantomData, path::PathBuf};

    use anyhow::{Ok, Result};

    use super::backend::{models::unet_2d, schedulers};
    use tch::{nn, Device, Kind, Tensor};

    pub(crate) trait UnetState {}

    pub(crate) struct UnetStateNew;
    impl UnetState for UnetStateNew {}
    pub(crate) struct UnetStateBuild;
    impl UnetState for UnetStateBuild {}
    pub(crate) struct UnetStateImage;
    impl UnetState for UnetStateImage {}
    pub(crate) struct UnetStateImageDone;
    impl UnetState for UnetStateImageDone {}

    pub(crate) struct Unet<T>
    where
        T: UnetState,
    {
        width: i64,
        height: i64,
        steps: usize,
        config: unet_2d::UNet2DConditionModelConfig,
        scheduler_config: schedulers::ddim::DDIMSchedulerConfig,
        device: Device,
        weights: PathBuf,
        model: Option<unet_2d::UNet2DConditionModel>,
        scheduler: Option<schedulers::ddim::DDIMScheduler>,
        embeddings: Option<Tensor>,
        image: Option<Tensor>,
        timesteps: Option<Vec<usize>>,
        timestep: Option<usize>,
        _state: PhantomData<T>,
    }

    impl<T: UnetState> Unet<T> {
        pub(crate) fn new(
            width: i64,
            height: i64,
            steps: usize,
            config: unet_2d::UNet2DConditionModelConfig,
            scheduler_config: schedulers::ddim::DDIMSchedulerConfig,
            device: Device,
            weights: PathBuf,
        ) -> Unet<UnetStateNew> {
            Unet {
                width,
                height,
                steps,
                config,
                scheduler_config,
                device,
                weights,
                model: None,
                scheduler: None,
                embeddings: None,
                image: None,
                timesteps: None,
                timestep: None,
                _state: PhantomData,
            }
        }
    }

    impl Unet<UnetStateNew> {
        pub(crate) fn build(self) -> Result<Unet<UnetStateBuild>> {
            let mut vs_unet = nn::VarStore::new(self.device);
            let unet =
                unet_2d::UNet2DConditionModel::new(vs_unet.root(), 4, 4, self.config.clone());
            vs_unet.load(self.weights.clone())?;

            let scheduler = schedulers::ddim::DDIMScheduler::new(self.steps, self.scheduler_config);

            Ok(Unet {
                width: self.width,
                height: self.height,
                steps: self.steps,
                config: self.config,
                scheduler_config: self.scheduler_config,
                device: self.device,
                weights: self.weights,
                model: Some(unet),
                scheduler: Some(scheduler),
                embeddings: None,
                image: None,
                timesteps: None,
                timestep: None,
                _state: PhantomData,
            })
        }
    }

    impl Unet<UnetStateBuild> {
        pub(crate) fn image(self, embeddings: &Tensor, seed: i64) -> Unet<UnetStateImage> {
            let scheduler = self.scheduler.as_ref().expect("scheduler not initialized");

            tch::manual_seed(seed);
            let mut image = Tensor::randn(
                &[1, 4, self.height / 8, self.width / 8],
                (Kind::Float, self.device),
            );

            image *= scheduler.init_noise_sigma();

            let timesteps = scheduler.timesteps().to_vec();

            Unet {
                width: self.width,
                height: self.height,
                steps: self.steps,
                config: self.config,
                scheduler_config: self.scheduler_config,
                device: self.device,
                weights: self.weights,
                model: self.model,
                scheduler: self.scheduler,
                embeddings: Some(embeddings.shallow_clone()),
                image: Some(image),
                timesteps: Some(timesteps),
                timestep: Some(0),
                _state: PhantomData,
            }
        }
    }

    impl Unet<UnetStateImage> {
        pub(crate) fn step(&mut self) -> Option<()> {
            let model = self.model.as_ref().expect("model not initialized");
            let scheduler = self.scheduler.as_ref().expect("scheduler not initialized");
            let embeddings = self
                .embeddings
                .as_ref()
                .expect("embeddings not initialized");
            let image = self.image.as_ref().expect("image not initialized");
            let timesteps = self.timesteps.as_ref().expect("timestep not initialized");
            let timestep = self.timestep.expect("timestep not initialized");

            let timestep_mapped = timesteps[timestep];

            let image_model_input = Tensor::cat(&[image, image], 0);

            let image_model_input = scheduler.scale_model_input(image_model_input, timestep_mapped);
            let noise_pred = model.forward(
                &image_model_input,
                timestep_mapped as f64,
                &embeddings.to(self.device),
            );
            let noise_pred = noise_pred.chunk(2, 0);
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            const GUIDANCE_SCALE: f64 = 7.5;
            let noise_pred =
                noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE;
            let image = scheduler.step(&noise_pred, timestep_mapped, &image);

            if timestep >= self.steps - 1 {
                self.image = Some(image);
                self.timesteps = None;
                return Some(());
            }

            self.timestep = Some(timestep + 1);
            self.image = Some(image);
            None
        }

        pub(crate) fn finish(self) -> Unet<UnetStateImageDone> {
            Unet {
                width: self.width,
                height: self.height,
                steps: self.steps,
                config: self.config,
                scheduler_config: self.scheduler_config,
                device: self.device,
                weights: self.weights,
                model: self.model,
                scheduler: self.scheduler,
                embeddings: self.embeddings,
                image: self.image,
                timesteps: None,
                timestep: None,
                _state: PhantomData,
            }
        }
    }

    impl Unet<UnetStateImageDone> {
        pub(crate) fn image(&self) -> Tensor {
            self.image
                .as_ref()
                .expect("image not initialized")
                .shallow_clone()
        }

        pub(crate) fn reset(self) -> Result<Unet<UnetStateBuild>> {
            Ok(Unet {
                width: self.width,
                height: self.height,
                steps: self.steps,
                config: self.config,
                scheduler_config: self.scheduler_config,
                device: self.device,
                weights: self.weights,
                model: self.model,
                scheduler: self.scheduler,
                embeddings: None,
                image: None,
                timesteps: None,
                timestep: None,
                _state: PhantomData,
            })
        }
    }
}
