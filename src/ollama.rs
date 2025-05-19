use std::time::Duration;

use anyhow::bail;
use log::debug;
use reqwest::{StatusCode, blocking::Client};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub struct Ollama {
    client: Client,
    ollama_url: String,
    model: String,
}

impl Ollama {
    pub fn new(ollama_url: String, model: String, timeout: Duration) -> anyhow::Result<Self> {
        let client = Client::builder().timeout(timeout).build()?;
        if let Err(e) = client
            .get(format!("{}/api/version", ollama_url))
            .send()
            .and_then(|r| r.error_for_status())
        {
            bail!("Unable to connect to ollama. Is it running?: {e}");
        }

        Ok(Self {
            client,
            ollama_url,
            model,
        })
    }

    pub fn model_exists(&self) -> anyhow::Result<bool> {
        let show_res = self
            .client
            .post(format!("{}/api/show", self.ollama_url))
            .json(&Model {
                name: self.model.clone(),
            })
            .send()
            .and_then(|r| r.error_for_status());

        let model_exists = match show_res {
            Ok(_) => {
                debug!("Model already present locally");
                true
            }
            Err(e) => {
                if let Some(StatusCode::NOT_FOUND) = e.status() {
                    false
                } else {
                    bail!("Error checking model: {e}")
                }
            }
        };
        Ok(model_exists)
    }

    pub fn pull(&self) -> anyhow::Result<()> {
        self.client
            .post(format!("{}/api/pull", self.ollama_url))
            // Two minutes should be enough for everyone..
            // Better to use the streaming mode to get continuous updates, but reqwest blocking don't expose the separate chunks, just a stream, so parsing would be a bit tricky
            .timeout(Duration::from_secs(120))
            .json(&PullParams {
                model: self.model.clone(),
                stream: false,
            })
            .send()
            .and_then(|r| r.error_for_status())?;
        Ok(())
    }

    pub fn chat(&self, messages: &[Message]) -> Result<String, Error> {
        let response: ChatResponse = self
            .client
            .post(format!("{}/api/chat", self.ollama_url))
            .json(&ChatParams {
                model: &self.model,
                stream: false,
                messages,
                format: None,
            })
            .send()?
            .json()?;
        if &response.message.role != "assistant" {
            return Err(Error::Role(response.message.role.clone()));
        }
        Ok(response.message.content)
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("HTTP error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("Invalid role: {0}")]
    Role(String),
}

#[derive(Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize)]
struct PullParams {
    model: String,
    stream: bool,
}

#[derive(Serialize)]
struct ChatParams<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    format: Option<&'a str>,
}

#[derive(Serialize, Deserialize)]
struct ChatResponse {
    message: Message,
}

#[derive(Serialize, Deserialize, Debug)]
struct Model {
    name: String,
}
