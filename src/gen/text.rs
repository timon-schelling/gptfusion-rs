use std::path::Path;

use anyhow::Result;

use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::LocalResource;

pub fn test() -> Result<()> {
    let model_path = Path::new("data/weights/text/gptneo/1-3b");
    //    Set-up model resources
    let config_resource = Box::new(LocalResource {
        local_path: model_path.join("config.json"),
    });
    let vocab_resource = Box::new(LocalResource {
        local_path: model_path.join("vocab.json"),
    });
    let merges_resource = Box::new(LocalResource {
        local_path: model_path.join("merges.txt"),
    });
    let model_resource = Box::new(LocalResource {
        local_path: model_path.join("model.ot"),
    });
    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPTNeo,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        min_length: 10,
        max_length: Some(32),
        do_sample: false,
        early_stopping: true,
        num_beams: 4,
        num_return_sequences: 1,
        device: tch::Device::Cpu,
        ..Default::default()
    };


    let mut model = TextGenerationModel::new(generate_config)?;
    model.half();
    model.set_device(tch::Device::cuda_if_available());

    tch::manual_seed(512);

    let input_context = "Random photo description:";
    let output = model.generate(&[input_context], None);

    for sentence in output {
        println!("{sentence}");
    }
    Ok(())
}
