use std::{
    path::PathBuf,
    sync::mpsc::{Receiver, Sender},
    time,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use tch::{Device, Tensor};

pub(crate) mod backend;

use self::backend::{pipelines::stable_diffusion::StableDiffusionConfig};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Config {
    pub width: i64,
    pub height: i64,
    pub steps: usize,
    pub acceleration_config: AccelerationConfig,
    pub version: StableDiffusionVersion,
    pub vocab_file: PathBuf,
    pub clip_weights: PathBuf,
    pub vae_weights: PathBuf,
    pub unet_weights: PathBuf,
    pub sliced_attention_size: Option<i64>,
}

impl Config {
    fn default(version: StableDiffusionVersion) -> Config {
        let mut base_path = PathBuf::from("data/weights/image/stable-diffusion");
        base_path.push(match version {
            StableDiffusionVersion::V1_5 => "v1-5",
            StableDiffusionVersion::V2_1 => "v2-1",
        });

        Self {
            width: 512,
            height: 512,
            steps: 30,
            acceleration_config: AccelerationConfig::default(),
            version,
            vocab_file: base_path.join("vocab.txt"),
            clip_weights: base_path.join("clip.ot").into(),
            vae_weights: base_path.join("vae.ot").into(),
            unet_weights: base_path.join("unet.ot").into(),
            sliced_attention_size: None,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::default(StableDiffusionVersion::V1_5)
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum StableDiffusionVersion {
    V1_5,
    V2_1,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
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
) -> StableDiffusionConfig {
    match version {
        StableDiffusionVersion::V1_5 => {
            StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::V2_1 => {
            StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum Workload {
    Clip,
    Vae,
    Unet,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AccelerationConfig {
    clip: bool,
    vae: bool,
    unet: bool,
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            clip: true,
            vae: false,
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

mod clip {
    use std::{marker::PhantomData, path::PathBuf};

    use anyhow::Result;

    use super::backend::transformers::clip;
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
        uncond_tokens: Option<Tensor>,
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
                uncond_tokens: None,
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

            Ok(Clip {
                config: self.config,
                device: self.device,
                vocab: self.vocab,
                weights: self.weights,
                model: Some(model),
                tokenizer: Some(tokenizer),
                uncond_tokens: self.uncond_tokens,
                _state: PhantomData,
            })
        }
    }

    impl Clip<ClipStateBuild> {
        pub(crate) fn embed(&mut self, text: &String) -> Result<Tensor> {
            let model = self.model.as_ref().expect("model not initialized");

            let uncond_tokens = match &self.uncond_tokens {
                Some(uncond_tokens) => uncond_tokens.shallow_clone(),
                None => {
                    let uncond_tokens = self.tokenize(&"".to_string())?;
                    self.uncond_tokens = Some(uncond_tokens.shallow_clone());
                    uncond_tokens
                }
            };
            let uncond_embeddings = model.forward(&uncond_tokens);

            let tokens = self.tokenize(text)?;
            let embeddings = model.forward(&tokens);

            let embeddings = Tensor::cat(&[uncond_embeddings, embeddings], 0).to(self.device);

            Ok(embeddings)
        }

        fn tokenize(&self, text: &String) -> Result<Tensor> {
            let tokenizer = self.tokenizer.as_ref().expect("tokenizer not initialized");

            let tokens = tokenizer.encode(&text)?;
            let tokens = tokens.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
            let tokens = Tensor::from_slice(&tokens).view((1, -1)).to(self.device);

            Ok(tokens)
        }
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
    use tch::{nn, Device, Kind, NoGradGuard, Tensor};

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
        timestep: Option<usize>,
        no_grad_guard: Option<NoGradGuard>,
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
                timestep: None,
                no_grad_guard: None,
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
                timestep: None,
                no_grad_guard: None,
                _state: PhantomData,
            })
        }
    }

    impl Unet<UnetStateBuild> {
        pub(crate) fn image(self, embeddings: &Tensor, seed: i64) -> Unet<UnetStateImage> {
            let scheduler = self.scheduler.as_ref().expect("scheduler not initialized");

            let no_grad_guard = tch::no_grad_guard();

            tch::manual_seed(seed);
            let mut image = Tensor::randn(
                &[1, 4, self.height / 8, self.width / 8],
                (Kind::Float, self.device),
            );

            image *= scheduler.init_noise_sigma();

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
                timestep: Some(0),
                no_grad_guard: Some(no_grad_guard),
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
            let timestep = self.timestep.expect("timestep not initialized");

            let image_model_input = Tensor::cat(&[image, image], 0);

            let image_model_input = scheduler.scale_model_input(image_model_input, timestep);
            let noise_pred = model.forward(&image_model_input, timestep as f64, &embeddings);
            let noise_pred = noise_pred.chunk(2, 0);
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            const GUIDANCE_SCALE: f64 = 7.5;
            let noise_pred =
                noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE;
            let image = scheduler.step(&noise_pred, timestep, &image);

            if timestep >= self.steps - 1 {
                self.image = Some(image);
                self.timestep = None;
                return Some(());
            }

            self.image = Some(image);
            self.timestep = Some(timestep + 1);
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
                timestep: None,
                no_grad_guard: self.no_grad_guard,
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
                timestep: None,
                no_grad_guard: None,
                _state: PhantomData,
            })
        }
    }
}

fn exec_batch(args: Receiver<Args>, config: Config, status: Sender<Status>) -> Result<()> {
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
    } = config.clone();

    tch::maybe_init_cuda();

    status.send(Status::System {
        cuda: tch::Cuda::is_available(),
        cudnn: tch::Cuda::cudnn_is_available(),
        mps: tch::utils::has_mps(),
    })?;

    let sd_config =
        build_stable_diffusion_config(version, sliced_attention_size, Some(width), Some(height));

    let clip_device = acceleration_config.build_device_for(Workload::Clip);
    let vae_device = acceleration_config.build_device_for(Workload::Vae);
    let unet_device = acceleration_config.build_device_for(Workload::Unet);

    let clip = clip::Clip::<clip::ClipStateNew>::new(
        sd_config.clip.clone(),
        clip_device,
        vocab_file,
        clip_weights,
    );
    let vae = vae::Vae::<vae::VaeStateNew>::new(sd_config.vae, vae_device, vae_weights);
    let unet = unet::Unet::<unet::UnetStateNew>::new(
        width,
        height,
        steps,
        sd_config.unet.clone(),
        sd_config.scheduler.clone(),
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
            UnetWithState::Build(unet) => UnetWithState::Image(unet.image(embeds, seed)),
            UnetWithState::Image(mut unet) => match unet.step() {
                Some(()) => UnetWithState::Done(unet.finish()),
                None => UnetWithState::Image(unet),
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
        }
    }

    Ok(())
}
