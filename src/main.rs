use ansi_stripper::AnsiStripReader;
use anyhow::bail;
use clap::Parser;
use colorgrad::{BlendMode, Gradient};
use log::{debug, info, warn};
use ollama::{Message, Ollama};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Write as _;
use std::io::{BufRead, BufReader};
use std::sync::LazyLock;
use std::time::Duration;
use termcolor::{ColorChoice, ColorSpec, WriteColor as _};

mod ansi_stripper;
mod ollama;

const SYSTEM_PROMPT: &str = "You are a developer log analyzer.
Given a sequence of lines of text. Determine if it looks like a log file or not.
If it looks like a log rate the last line by how interesting you think it is for diagnosing an issue with the system.
Rate only the last line. Use the prior lines only for context.
If a prior line looks unrelated to the last one, disregard it.
Output EXACTLY in this format:
```
Very brief single-sentence analysis on a single line
SCORE: 0-100
```

If it doesn't look like a log file just respond with:
```
Not a log
SCORE: 0
```

Do NOT include any code examples, snippets, or additional explanations.
Keep responses strictly limited to the analysis and score.
Do NOT include any additional framing such as ```.
Do NOT start the analysis with \"The last line\" or similar redundant information.

Score guide:
Low (0-30): Routine/minor info
Medium (31-70): Noteworthy/important
High (71-100): Critical/security issues
";

const MODEL: &str = "hf.co/jnises/gemma-3-1b-llmog-GGUF:Q8_0";

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

    /// Request timeout in seconds
    ///
    /// Increase if you have slow hardware, long lines or a larger model.
    #[arg(long, default_value = "10")]
    timeout: u64,
}

#[derive(Serialize, Deserialize)]
struct LogScore {
    reasoning: String,
    score: f64,
}

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

    let ol = Ollama::new(
        cli.ollama_url,
        MODEL.to_string(),
        Duration::from_secs(cli.timeout),
    )
    .map_err(|e| {
        if e.is_connection_error() {
            anyhow::anyhow!(
                "Failed to connect to ollama at {}. Is ollama running?: {}",
                e.url().unwrap_or("unknown"),
                e
            )
        } else {
            anyhow::anyhow!("Failed to initialize ollama client: {}", e)
        }
    })?;

    let model_exists = ol.model_exists().map_err(|e| {
        anyhow::anyhow!(
            "Failed to check if model '{}' exists on ollama server at {}: {}",
            e.model().unwrap_or("unknown"),
            e.url().unwrap_or("unknown"),
            e
        )
    })?;

    if !model_exists {
        info!("Model '{MODEL}' not found locally, pulling...");
    }
    //  we pull even if the model exists in case it has been updated
    if let Err(e) = ol.pull() {
        if model_exists {
            debug!(
                "Unable to pull model '{}' from {}: {}. But it was already downloaded so that's ok",
                e.model().unwrap_or(MODEL),
                e.url().unwrap_or("unknown"),
                e
            );
        } else {
            bail!(
                "Unable to pull model '{}' from {}: {}",
                e.model().unwrap_or(MODEL),
                e.url().unwrap_or("unknown"),
                e
            );
        }
    }

    let reader = BufReader::new(AnsiStripReader::new(std::io::stdin().lock()));

    let response_re = regex::Regex::new(
        r"(?i)^(?:```\s*\n?)?(?P<reason>(?:.*\n)*?)SCORE:\s*(?P<score>\d+(?:\.\d+)?)\s*(?:```\s*\n?)?$",
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

        const MAX_RETRIES: usize = 5;
        const MAX_TIMEOUTS: usize = 1;
        for retry in 0.. {
            let response = match ol.chat(&messages) {
                Ok(r) => r,
                Err(e) => {
                    // Handle timeout errors specifically
                    if e.is_timeout() {
                        if retry >= MAX_TIMEOUTS {
                            warn!(
                                "Too many timeouts communicating with ollama at {} for model '{}' (timeout: {}s)",
                                e.url().unwrap_or("unknown"),
                                e.model().unwrap_or("unknown"),
                                cli.timeout
                            );
                            so.reset()?;
                            writeln!(so, "{line}")?;
                            break;
                        } else {
                            debug!(
                                "Timeout communicating with ollama at {} for model '{}' (timeout: {}s)",
                                e.url().unwrap_or("unknown"),
                                e.model().unwrap_or("unknown"),
                                cli.timeout
                            );
                            continue;
                        }
                    }
                    // Handle connection errors
                    else if e.is_connection_error() {
                        warn!(
                            "Connection error communicating with ollama at {}: {}",
                            e.url().unwrap_or("unknown"),
                            e
                        );
                        bail!(e);
                    }
                    // Handle other errors
                    else {
                        bail!(e);
                    }
                }
            };

            // Extract the reason and score from the response
            if let Some((mut reason, score)) = response_re.captures(&response).and_then(|caps| {
                Some((
                    caps.name("reason")
                        .map_or_else(String::new, |m| m.as_str().to_string()),
                    caps["score"].parse::<f64>().ok()?,
                ))
            }) {
                let reason_lines: Vec<_> = reason.lines().collect();
                if reason_lines.len() > 1 && retry == 0 {
                    // If this is the first attempt and we got a multi-line reason, retry
                    // if it keeps giving us multiple lines, just pick the last one to avoid too many retries
                    debug!(
                        "First attempt returned multi-line reason, retrying: {}",
                        response
                    );
                    continue;
                }
                reason = reason_lines.last().unwrap_or(&"").trim().to_string();

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
            } else if retry >= MAX_RETRIES {
                warn!("Bad response from model: {response}");
                so.reset()?;
                writeln!(so, "{line}")?;
            } else {
                debug!("Retrying bad response from model: {response}");
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
