use anyhow::{Context, Result};
use hf_hub::{api::tokio::Api as HfApi, Repo as HfRepo, RepoType as HfRepoType};
use mistralrs::{
    ChatCompletionResponse, MistralRs, Request as MistralRequest, RequestMessage as MistralRequestMessage,
    SamplingParams, TextMessage as MistralTextMessage, TextMessageRole as MistralTextMessageRole,
    GgufModelBuilder, // Assuming this is the correct builder from v0_4_api
    Response, // For handling the output of send_chat_request_atomic if used
};
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;
use tokio::time::timeout as tokio_timeout;

#[derive(Error, Debug)]
pub enum MistralLocalError {
    #[error("Hugging Face Hub API error: {0}")]
    HfHubError(#[from] hf_hub::Error),
    #[error("MistralRS engine error: {0}")]
    MistralRsError(#[from] mistralrs::MistralRsError),
    #[error("Model download timed out after {0:?}")]
    DownloadTimeout(Duration),
    #[error("Inference request timed out after {0:?}")]
    InferenceTimeout(Duration),
    #[error("Invalid input: {0}")]
    InputError(String),
    #[error("Failed to build MistralRS engine: {0}")]
    EngineBuildError(String),
    #[error("No content in model response")]
    NoResponseContent,
}

// Simple struct for messages coming from main.rs
#[derive(Clone, Debug)]
pub struct UserMessage {
    pub role: String, // "user", "system", "assistant"
    pub content: String,
}

pub struct MistralLocalEngine {
    mistral_rs_engine: MistralRs,
    sampling_params: SamplingParams,
}

impl MistralLocalEngine {
    pub async fn new(
        model_repo_id: String,
        model_filename: String,
        tokenizer_filename: Option<String>,
        // chat_template_filename: Option<String>, // mistral.rs GGUF loader might infer this or use internal
        gguf_log_level: mistralrs::gguf::LogLevel, // e.g. mistralrs::gguf::LogLevel::Silence
        cache_dir: Option<PathBuf>,
        download_timeout: Duration,
    ) -> Result<Self, MistralLocalError> {
        let api = if let Some(cache) = cache_dir {
            HfApi::new_with_cache(cache).context("Failed to initialize HF API with custom cache")?
        } else {
            HfApi::new().context("Failed to initialize HF API")?
        };

        let repo = HfRepo::new(model_repo_id, HfRepoType::Model);
        let api_repo = api.repo(repo);

        let model_path = tokio_timeout(download_timeout, api_repo.get(&model_filename))
            .await
            .map_err(|_| MistralLocalError::DownloadTimeout(download_timeout))?
            .context(format!("Failed to download/locate model file: {}", model_filename))?;

        let mut tokenizer_path: Option<PathBuf> = None;
        if let Some(t_filename) = tokenizer_filename {
            tokenizer_path = Some(
                tokio_timeout(download_timeout, api_repo.get(&t_filename))
                    .await
                    .map_err(|_| MistralLocalError::DownloadTimeout(download_timeout))?
                    .context(format!("Failed to download/locate tokenizer file: {}", t_filename))?,
            );
        }
        
        // GgufModelBuilder is part of mistralrs::v0_4_api, ensure it's correctly pathed if used directly
        // For simplicity, assuming direct use here. May need mistralrs::v0_4_api::GgufModelBuilder
        let mut builder = GgufModelBuilder::new(
            /* model_id: */ None, // We are providing paths directly
            /* quantized_model_id: */ None,
            /* quantized_filename: */ vec![model_path.to_string_lossy().into_owned()],
            /* tokenizer_json: */ tokenizer_path.map(|p| p.to_string_lossy().into_owned()),
            /* chat_template: */ None, // Let mistral.rs try to infer or use its default for GGUF
        );

        builder = builder.with_gguf_log_level(gguf_log_level);
        // Add other builder configurations if necessary, e.g., .with_seed(), .with_sample_len()

        let mistral_rs_engine = builder
            .build()
            .await
            .map_err(|e| MistralLocalError::EngineBuildError(e.to_string()))?;
            
        // Define default sampling parameters
        // The prompt asks for deterministic output, so temperature 0.0
        let sampling_params = SamplingParams {
            temperature: Some(0.0),
            top_k: None,
            top_p: None,
            top_n_logprobs: 0,
            repeat_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
            min_p: None, // Added in mistralrs 0.5.0
            stop_toks: None,
            max_len: None, // Let the model decide or configure if needed
            logits_bias: None,
            ignore_eos: false, // Added in mistralrs 0.5.0
        };

        Ok(Self {
            mistral_rs_engine,
            sampling_params,
        })
    }

    pub async fn chat(
        &self,
        messages: Vec<UserMessage>,
        request_timeout: Duration,
    ) -> Result<String, MistralLocalError> {
        let mut mistral_messages = Vec::new();
        for msg in messages {
            let role = match msg.role.to_lowercase().as_str() {
                "user" => MistralTextMessageRole::User,
                "system" => MistralTextMessageRole::System,
                "assistant" => MistralTextMessageRole::Assistant,
                _ => return Err(MistralLocalError::InputError(format!("Unknown role: {}", msg.role))),
            };
            mistral_messages.push(MistralTextMessage {
                role,
                content: msg.content,
                tool_calls: None, // Not used in this application
                finish_reason: None, // Not applicable for input
            });
        }

        let chat_request = MistralRequest {
            messages: MistralRequestMessage::Chat(mistral_messages),
            sampling_params: self.sampling_params.clone(),
            response_format: None, // Default response format
            tools: None, // No tools needed
            tool_choice: None, // No tools needed
            prompt_chunk_size: None, // Default chunk size
            kind: (), // Placeholder for NormalRequest type param, may need adjustment
        };
        
        // The send_chat_request_atomic method is simpler if available and suitable
        // For more control, send_request is used.
        // The `MistralRs::send_chat_request` method from the example seems to be a helper.
        // Let's try to use the more direct `send_request` if `send_chat_request` isn't found on `MistralRs` directly
        // or if it doesn't fit the `MistralRequest` structure.
        // Based on mistralrs docs, `MistralRs` itself has `send_chat_request(TextMessages)`
        // Let's adapt to use TextMessages which seems simpler than raw MistralRequest
        
        let mut text_messages_for_engine = mistralrs::TextMessages::new();
        for msg in &chat_request.messages { // Assuming chat_request.messages is Vec<MistralTextMessage>
             match msg {
                MistralRequestMessage::Chat(m_msgs) => {
                    for m_msg in m_msgs {
                         text_messages_for_engine.add_message(m_msg.role, m_msg.content.clone());
                    }
                }
                _ => return Err(MistralLocalError::InputError("Unsupported message type for chat".to_string())),
             }
        }


        let response_result = tokio_timeout(
            request_timeout,
            self.mistral_rs_engine.send_chat_request(text_messages_for_engine, self.sampling_params.clone(), None),
        )
        .await;

        match response_result {
            Ok(Ok(Response::Model(ChatCompletionResponse { choices, .. }))) | Ok(Ok(Response::Done(ChatCompletionResponse { choices, .. }))) => {
                if let Some(choice) = choices.first() {
                    if let Some(content) = &choice.message.content {
                        Ok(content.clone())
                    } else {
                        Err(MistralLocalError::NoResponseContent)
                    }
                } else {
                    Err(MistralLocalError::NoResponseContent)
                }
            }
            Ok(Ok(other_response)) => {
                 Err(MistralLocalError::InputError(format!("Unexpected response type from model: {:?}", other_response)))
            }
            Ok(Err(e)) => Err(MistralLocalError::MistralRsError(e)),
            Err(_) => Err(MistralLocalError::InferenceTimeout(request_timeout)),
        }
    }
}
