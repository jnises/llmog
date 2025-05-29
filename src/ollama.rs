use std::time::Duration;

use anyhow;
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
    pub fn new(ollama_url: String, model: String, timeout: Duration) -> Result<Self, Error> {
        let client = Client::builder().timeout(timeout).build()
            .map_err(|e| Error::Connection { url: ollama_url.clone(), source: e })?;
        
        if let Err(e) = client
            .get(format!("{}/api/version", ollama_url))
            .send()
            .and_then(|r| r.error_for_status())
        {
            return Err(Error::Connection { url: ollama_url, source: e });
        }

        Ok(Self {
            client,
            ollama_url,
            model,
        })
    }

    pub fn model_exists(&self) -> Result<bool, Error> {
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
                    return Err(Error::ModelCheck { 
                        model: self.model.clone(), 
                        url: self.ollama_url.clone(), 
                        source: e 
                    });
                }
            }
        };
        Ok(model_exists)
    }

    /// Ensures the model exists, pulling it if necessary
    pub fn ensure_model(&self) -> Result<(), Error> {
        if !self.model_exists()? {
            return Err(Error::ModelNotFound {
                model: self.model.clone(),
                url: self.ollama_url.clone(),
            });
        }
        Ok(())
    }

    pub fn pull(&self) -> Result<(), Error> {
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
            .and_then(|r| r.error_for_status())
            .map_err(|e| Error::ModelPull { 
                model: self.model.clone(), 
                url: self.ollama_url.clone(), 
                source: e 
            })?;
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
            .send()
            .map_err(|e| Error::ChatRequest { 
                model: self.model.clone(), 
                url: self.ollama_url.clone(), 
                source: e 
            })?
            .json()
            .map_err(|e| Error::ChatRequest { 
                model: self.model.clone(), 
                url: self.ollama_url.clone(), 
                source: e 
            })?;
        if &response.message.role != "assistant" {
            return Err(Error::Role { role: response.message.role.clone() });
        }
        Ok(response.message.content)
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("HTTP error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("Invalid role: expected 'assistant', got '{role}'")]
    Role { role: String },

    #[error("Failed to connect to ollama at {url}: {source}")]
    Connection { url: String, source: reqwest::Error },

    #[error("Model '{model}' not found on ollama server at {url}")]
    ModelNotFound { 
        model: String, 
        url: String,
    },

    #[error("Failed to pull model '{model}' from ollama server at {url}: {source}")]
    ModelPull { 
        model: String, 
        url: String, 
        source: reqwest::Error 
    },

    #[error("Chat request failed for model '{model}' on ollama server at {url}: {source}")]
    ChatRequest { 
        model: String, 
        url: String, 
        source: reqwest::Error 
    },

    #[error("Failed to check if model '{model}' exists on ollama server at {url}: {source}")]
    ModelCheck { 
        model: String, 
        url: String, 
        source: reqwest::Error 
    },
}

impl Error {
    /// Returns true if this error is a timeout error
    pub fn is_timeout(&self) -> bool {
        match self {
            Error::Reqwest(e) | 
            Error::Connection { source: e, .. } |
            Error::ModelCheck { source: e, .. } |
            Error::ModelPull { source: e, .. } |
            Error::ChatRequest { source: e, .. } => e.is_timeout(),
            _ => false,
        }
    }

    /// Returns true if this error is a connection-related error
    pub fn is_connection_error(&self) -> bool {
        match self {
            Error::Connection { .. } => true,
            Error::Reqwest(e) |
            Error::ModelCheck { source: e, .. } |
            Error::ModelPull { source: e, .. } |
            Error::ChatRequest { source: e, .. } => e.is_connect(),
            _ => false,
        }
    }

    /// Returns the model name if this error is model-related
    pub fn model(&self) -> Option<&str> {
        match self {
            Error::ModelNotFound { model, .. } |
            Error::ModelPull { model, .. } |
            Error::ChatRequest { model, .. } |
            Error::ModelCheck { model, .. } => Some(model),
            _ => None,
        }
    }

    /// Returns the ollama URL if available
    pub fn url(&self) -> Option<&str> {
        match self {
            Error::Connection { url, .. } |
            Error::ModelNotFound { url, .. } |
            Error::ModelPull { url, .. } |
            Error::ChatRequest { url, .. } |
            Error::ModelCheck { url, .. } => Some(url),
            _ => None,
        }
    }

    /// Converts this error to an anyhow error with additional context
    pub fn to_anyhow_with_context(self, context: &str) -> anyhow::Error {
        anyhow::anyhow!("{}: {}", context, self)
    }
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
