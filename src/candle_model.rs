// src/candle_model.rs
use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor, DType};
use candle_transformers::models::quantized_gemma3; // Adjust if model is not Gemma3 or path is different
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use std::path::PathBuf;
use std::fs::File; // For reading GGUF metadata if needed directly

// May need these for GGUF loading
use candle_core::quantized::{gguf_file, QMatMul, QuantizedLoader, GgmlType};
use candle_transformers::quantized_var_builder::QuantizedVarBuilder;


const MODEL_REPO: &str = "jnises/gemma-3-1b-llmog-GGUF";
// Assuming the GGUF file is named this within the repo. This might need adjustment.
const GGUF_FILENAME: &str = "model-Q8_0.gguf"; 
const TOKENIZER_FILENAME: &str = "tokenizer.json";

pub struct CandleModel {
    model: quantized_gemma3::Model, // Or the appropriate quantized model type
    tokenizer: Tokenizer,
    device: Device,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl CandleModel {
    pub fn new(
        model_repo_id: Option<String>,
        gguf_filename: Option<String>,
        tokenizer_filename: Option<String>,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device_type: Device, // Allow specifying CPU/GPU
    ) -> Result<Self> {
        let device = device_type;
        let api = Api::new()?;
        let repo_id = model_repo_id.unwrap_or_else(|| MODEL_REPO.to_string());
        let repo = api.repo(Repo::with_revision(repo_id, RepoType::Model, "main".to_string()));

        let gguf_path = repo.get(&gguf_filename.unwrap_or_else(|| GGUF_FILENAME.to_string()))?;
        let tokenizer_path = repo.get(&tokenizer_filename.unwrap_or_else(|| TOKENIZER_FILENAME.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);

        // Load GGUF model
        let mut file = File::open(&gguf_path)?;
        let gguf_content = gguf_file::Content::read(&mut file).map_err(|e| E::msg(format!("Failed to read GGUF file: {e}")))?;
        
        // Derive config from GGUF metadata
        // This is a simplified version. A more robust solution would involve mapping more metadata fields.
        let config = quantized_gemma3::Config::gemma_3_1b(tokenizer.get_vocab_size(true));
        // TODO: Further populate config from gguf_content.metadata if necessary.
        // For example:
        // config.num_hidden_layers = gguf_content.metadata.get("gemma.block_count").and_then(|v| v.to_u32().ok()).unwrap_or(config.num_hidden_layers);
        // config.hidden_size = gguf_content.metadata.get("gemma.embedding_length").and_then(|v| v.to_u32().ok()).unwrap_or(config.hidden_size);
        // ... and so on for other relevant fields.

        let vb = QuantizedVarBuilder::from_gguf(gguf_path, &device)?;
        let model = quantized_gemma3::Model::new(&config, vb)?; 

        Ok(Self {
            model,
            tokenizer,
            device,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
        })
    }

    pub fn chat(&mut self, system_prompt: &str, user_prompt: &str, max_tokens: usize) -> Result<String> {
        // Gemma instruct format:
        // The system prompt is part of the first user turn.
        let full_prompt = format!("<start_of_turn>user
{system_prompt}

{user_prompt}<end_of_turn>
<start_of_turn>model
");

        // self.tokenizer.clear(); // Tokenizer does not have a clear method. Re-encoding should be fine.
        let mut tokens = self.tokenizer.encode(full_prompt, true).map_err(E::msg)?.get_ids().to_vec();

        let mut generated_tokens = Vec::new();
        // TODO: Confirm EOS token for this specific model/tokenizer.
        // Gemma's official EOS is <eos> (id 1), but some tokenizers might use <end_of_turn> or other tokens.
        // Using <end_of_turn> as it's explicitly mentioned in the prompt format.
        let eos_token_str = "<end_of_turn>"; 
        let eos_token = self.tokenizer.token_to_id(eos_token_str).ok_or_else(|| E::msg(format!("EOS token '{eos_token_str}' not found in tokenizer")))?;


        for i in 0..max_tokens {
            let context_size = if i > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let input_tensor = Tensor::new(&tokens[start_pos..], &self.device)?.unsqueeze(0)?;
            
            let logits = self.model.forward(&input_tensor, start_pos)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            
            let penalized_logits = if self.repeat_penalty == 1.0 {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&penalized_logits)?;
            tokens.push(next_token);
            generated_tokens.push(next_token);

            if next_token == eos_token {
                break;
            }
        }
        
        let generated_text = self.tokenizer.decode(&generated_tokens, true).map_err(E::msg)?;
        Ok(generated_text)
    }
}
