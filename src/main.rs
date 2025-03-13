use ansi_stripper::AnsiStripReader;
use anyhow::bail;
use clap::Parser;
use colorgrad::{BlendMode, Gradient};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Write as _;
use std::io::{BufRead, BufReader};
use std::sync::LazyLock;
use termcolor::{ColorChoice, ColorSpec, WriteColor as _};
use ureq::Agent;

mod ansi_stripper;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Show analysis after each line
    #[arg(long)]
    analysis: bool,

    /// Ollama API URL
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,
}

#[derive(Serialize, Deserialize)]
struct PullParams {
    model: String,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct ChatParams {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    format: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize)]
struct ChatResponse {
    message: Message,
}

#[derive(Serialize, Deserialize)]
struct LogScore {
    reasoning: String,
    score: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct Model {
    name: String,
}

const SYSTEM_PROMPT: &str =
    "You are a developer log analyzer. Rate each log line by uniqueness, impact, and actionability.
For each line, output EXACTLY in this format:
```
[Very brief single-sentence analysis on a single line]
SCORE: [0-100]
```

Do NOT include any code examples, snippets, or additional explanations.
Keep responses strictly limited to the analysis and score.

Score guide:
Low (0-30): Routine/minor info
Medium (31-70): Noteworthy/important
High (71-100): Critical/security issues";

const MODEL: &str = "llama3.2";
//const MODEL: &str = "gemma3:4b";

// TODO: make this a cli argument
const LINE_WINDOW: usize = 3;

static GRADIENT: LazyLock<colorgrad::LinearGradient> = LazyLock::new(|| {
    colorgrad::GradientBuilder::new()
        .colors(&[
            colorgrad::Color::new(0.5, 0.5, 0.5, 1.0),
            colorgrad::Color::new(1.0, 1.0, 0.0, 1.0),
            colorgrad::Color::new(1.0, 0.0, 0.0, 1.0),
        ])
        .domain(&[0.3, 0.7, 1.0])
        .mode(BlendMode::Oklab)
        .build()
        .unwrap()
});

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let agent = Agent::new_with_defaults();
    // Check if model is already available locally
    let show_res = agent
        .post(format!("{}/api/show", cli.ollama_url))
        .send_json(Model {
            name: MODEL.to_string(),
        });

    let model_exists = match show_res {
        Ok(_) => true,
        Err(ureq::Error::StatusCode(404)) => false,
        Err(e) => bail!("Error checking model: {e}"),
    };

    if !model_exists {
        info!("Model '{MODEL}' not found locally, pulling...");
    }

    if let Err(e) = agent
        .post(format!("{}/api/pull", cli.ollama_url))
        .send_json(PullParams {
            model: MODEL.to_string(),
            stream: false,
        })
    {
        bail!("Unable to connect to ollama: {e}");
    }
    let reader = BufReader::new(AnsiStripReader::new(std::io::stdin().lock()));

    let response_re =
        regex::Regex::new(r"(?is)^(?:(?P<reason>.*?)\n)?\s*SCORE:\s*(?P<score>\d+(?:\.\d+)?)\s*$")?;

    let mut so = termcolor::BufferedStandardStream::stdout(ColorChoice::Auto);
    struct Exchange {
        request: String,
        response: String,
    }
    let mut history: VecDeque<Exchange> = VecDeque::new();
    for line in reader.lines() {
        so.flush()?;
        let line = line?;
        if line.trim().is_empty() {
            writeln!(so)?;
            continue;
        }

        for retry in 0.. {
            let mut messages = Vec::with_capacity(history.len() + 2);
            messages.push(Message {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            });
            for Exchange { request, response } in &history {
                messages.push(Message {
                    role: "user".to_string(),
                    content: request.clone(),
                });
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: response.clone(),
                });
            }
            messages.push(Message {
                role: "user".to_string(),
                content: line.clone(),
            });
            let response: ChatResponse = agent
                .post(format!("{}/api/chat", cli.ollama_url))
                .send_json(ChatParams {
                    model: MODEL.to_string(),
                    stream: false,
                    messages,
                    format: None,
                })?
                .body_mut()
                .read_json()?;
            if &response.message.role != "assistant" {
                anyhow::bail!("bad reponse role");
            }

            if let Some((reason, score)) =
                response_re
                    .captures(&response.message.content)
                    .and_then(|caps| {
                        Some((
                            caps.name("reason")
                                .map_or_else(String::new, |m| m.as_str().trim().to_string()),
                            caps["score"].parse::<f64>().ok()?,
                        ))
                    })
            {
                let color = GRADIENT.at((score.clamp(0.0, 100.0) / 100.0) as f32);
                let mut color_spec = ColorSpec::new();
                color_spec.set_fg(Some(colorgrad_to_term(color)));
                so.set_color(&color_spec)?;
                write!(so, "{line}")?;
                if cli.analysis {
                    so.reset()?;
                    write!(so, " : {reason}")?;
                }
                writeln!(so)?;
                if history.len() > LINE_WINDOW {
                    history.pop_front();
                }
                history.push_back(Exchange {
                    request: line,
                    response: response.message.content,
                });
            } else if retry > 10 {
                warn!("Bad response from model: {}", response.message.content);
                so.reset()?;
                writeln!(so, "{line}")?;
            } else {
                debug!(
                    "Retrying bad response from model: {}",
                    response.message.content
                );
                continue;
            }

            break;
        }
    }
    Ok(())
}

fn colorgrad_to_term(c: colorgrad::Color) -> termcolor::Color {
    let [r, g, b, _] = c.to_rgba8();
    termcolor::Color::Rgb(r, g, b)
}
