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
use std::time::Duration;
use termcolor::{ColorChoice, ColorSpec, WriteColor as _};
use ureq::Agent;
use ureq::config::Config;

mod ansi_stripper;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Show analysis after each line
    #[arg(long)]
    analyze: bool,

    /// Ollama API URL
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Number of lines to use for context
    #[arg(long, default_value = "3")]
    context: usize,
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

const SYSTEM_PROMPT: &str = "You are a developer log analyzer.
Given a sequence of log lines. Rate only the last line. Use the prior lines only for context.
If a prior line looks unrelated to the last one, disregard it.
Rate the last line by how interesting you think it is for diagnosing an issue with the system.
Output EXACTLY in this format:
```
Very brief single-sentence analysis on a single line
SCORE: 0-100
```

Do NOT include any code examples, snippets, or additional explanations.
Keep responses strictly limited to the analysis and score.
Do NOT include any additional framing such as ````.
Do NOT start the analysis with \"The last line\" or similar redundant information.

Score guide:
Low (0-30): Routine/minor info
Medium (31-70): Noteworthy/important
High (71-100): Critical/security issues
";

const MODEL: &str = "hf.co/jnises/gemma-3-1b-llmog-GGUF:Q8_0";

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

    let agent = Agent::new_with_config(
        Config::builder()
            .timeout_connect(Some(Duration::from_secs(20)))
            .build(),
    );
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
    //  we pull even if the model exists in case it has been updated
    if let Err(e) = agent
        .post(format!("{}/api/pull", cli.ollama_url))
        .send_json(PullParams {
            model: MODEL.to_string(),
            stream: false,
        })
    {
        if model_exists {
            debug!("Unable to pull model. But it was already downloaded so that's ok: {e}");
        } else {
            bail!("Unable to connect to ollama: {e}");
        }
    }

    let reader = BufReader::new(AnsiStripReader::new(std::io::stdin().lock()));

    let response_re = regex::Regex::new(
        r"(?i)^(?:```\s*\n?)?(?:(?P<reason>.*?)\n)?\s*SCORE:\s*(?P<score>\d+(?:\.\d+)?)\s*(?:```\s*\n?)?$",
    )?;

    let mut so = termcolor::BufferedStandardStream::stdout(ColorChoice::Auto);
    let mut history: VecDeque<String> = VecDeque::new();
    for line in reader.lines() {
        so.flush()?;
        let line = line?;
        debug_assert!(!line.ends_with('\n'));
        if history.len() >= cli.context {
            history.pop_front();
        }
        history.push_back(line.clone());
        if line.trim().is_empty() {
            writeln!(so)?;
            continue;
        }

        let mut line_concat = String::new();
        for line in &history {
            line_concat.push_str(line);
            line_concat.push('\n');
        }
        let messages = [
            Message {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: line_concat,
            },
        ];

        for retry in 0.. {
            // TODO: don't stop processing just because of temporary errors
            let response: ChatResponse = agent
                .post(format!("{}/api/chat", cli.ollama_url))
                .send_json(ChatParams {
                    model: MODEL,
                    stream: false,
                    messages: &messages,
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
                if cli.analyze {
                    let mut analyze_color_spec = ColorSpec::new();
                    analyze_color_spec.set_italic(true);
                    so.set_color(&analyze_color_spec)?;
                    write!(so, " » {reason}")?;
                }
                writeln!(so)?;
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
