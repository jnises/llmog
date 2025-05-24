use ansi_stripper::AnsiStripReader;
use anyhow::{bail, Context}; // Ensure Context is imported
use clap::Parser;
use colorgrad::{BlendMode, Gradient};
use log::{debug, error, info, warn}; // Ensure error is imported
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Write as _;
use std::io::{BufRead, BufReader};
use std::path::PathBuf; // Added for model_cache_dir
use std::sync::LazyLock;
use std::time::Duration;
use termcolor::{ColorChoice, ColorSpec, WriteColor as _};

mod ansi_stripper;
mod mistralrs_local; // Added
use mistralrs_local::{MistralLocalEngine, MistralLocalError, UserMessage}; // Added

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

// const MODEL: &str = "hf.co/jnises/gemma-3-1b-llmog-GGUF:Q8_0"; // Removed

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Show analysis after each line
    #[arg(long)]
    analyze: bool,

    /// Optional: Log level for GGUF loader (trace, debug, info, warn, error, silence). Default: silence
    #[arg(long, default_value = "silence")]
    gguf_log_level: String,

    /// Optional: Custom cache directory for Hugging Face models.
    #[arg(long)]
    model_cache_dir: Option<PathBuf>,

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let gguf_log_level_str = cli.gguf_log_level.to_lowercase();
    let gguf_log_level = match gguf_log_level_str.as_str() {
        "trace" => mistralrs::gguf::LogLevel::Trace,
        "debug" => mistralrs::gguf::LogLevel::Debug,
        "info" => mistralrs::gguf::LogLevel::Info,
        "warn" => mistralrs::gguf::LogLevel::Warn,
        "error" => mistralrs::gguf::LogLevel::Error,
        "silence" => mistralrs::gguf::LogLevel::Silence,
        _ => {
            warn!(
                "Invalid GGUF log level '{}'. Defaulting to Silence.",
                cli.gguf_log_level
            );
            mistralrs::gguf::LogLevel::Silence
        }
    };

    info!("Initializing MistralLocalEngine with the default pre-configured model.");
    let engine = MistralLocalEngine::new(
        gguf_log_level,
        cli.model_cache_dir.clone(),
        Duration::from_secs(cli.timeout * 3), // Increased timeout for potential model downloads
    )
    .await
    .context("Failed to initialize MistralLocalEngine")?;
    info!("MistralLocalEngine initialized successfully.");

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
        for hist_line in &history { // Renamed inner variable to avoid conflict if needed
            line_concat.push_str(hist_line);
            line_concat.push('\n');
        }

        let user_messages = vec![
            UserMessage {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            },
            UserMessage {
                role: "user".to_string(),
                content: line_concat.clone(), // Use accumulated history
            },
        ];

        const MAX_RETRIES: usize = 5;
        // const MAX_TIMEOUTS: usize = 1; // Removed

        for retry in 0.. {
            let chat_result = engine
                .chat(user_messages.clone(), Duration::from_secs(cli.timeout))
                .await;

            let response = match chat_result {
                Ok(r) => r,
                Err(e) => {
                    match e {
                        MistralLocalError::InferenceTimeout(d) => {
                            warn!("Inference for line '{}' timed out after {:?}", line, d);
                        }
                        MistralLocalError::MistralRsError(m_err) => {
                            warn!("MistralRs engine error for line '{}': {}", line, m_err);
                        }
                        MistralLocalError::NoResponseContent => {
                            // This will make the response an empty string,
                            // causing regex match to fail and trigger retry logic.
                            debug!("No content in model response for line '{}'. Retrying if possible.", line);
                            String::new() 
                        }
                        _ => {
                            error!(
                                "Fatal error processing line '{}': {:?}. Check model configuration or engine state.",
                                line, e
                            );
                            so.reset()?;
                            writeln!(so, "{line}")?;
                            break; // Break from the retry loop for this line
                        }
                    };
                    // If the error was InferenceTimeout or MistralRsError, also print line and break retry.
                    if matches!(e, MistralLocalError::InferenceTimeout(_) | MistralLocalError::MistralRsError(_)) {
                        so.reset()?;
                        writeln!(so, "{line}")?;
                        break; // Break from the retry loop for this specific line
                    }
                    // If it was NoResponseContent, `response` is String::new() and will be handled by regex logic.
                    // If it was a fatal error (the `_` arm), we already broke.
                    // If we reach here due to NoResponseContent, `response` is empty.
                    // Otherwise, if we broke, this assignment isn't strictly needed but harmless.
                    if !matches!(e, MistralLocalError::NoResponseContent) { // Avoid reassigning if it was NoResponseContent
                        // This path should ideally not be taken if a break occurred.
                        // If an error occurred that implies we can't proceed with regex parsing
                        error!("Failed to get usable chat response for line '{}': {:?}", line, e);
                        so.reset()?;
                        writeln!(so, "{line}")?;
                        break; // Break from retry loop
                    }
                    // If NoResponseContent, response is already String::new()
                    // Fall through to regex check which will fail and trigger retry/max_retries logic
                    String::new() // Ensure response is assigned for the NoResponseContent case if not already
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
                    debug!(
                        "First attempt for line '{}' returned multi-line reason, retrying: {}",
                        line, response
                    );
                    continue;
                }
                reason = reason_lines.last().unwrap_or(&"").trim().to_string();
                if reason.is_empty() && score == 0.0 && response.to_lowercase().contains("not a log") {
                     reason = "Not a log".to_string(); // Normalize "Not a log"
                }


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
                break; // Break from retry loop on success
            } else if retry >= MAX_RETRIES {
                warn!(
                    "Bad response from model (max retries reached) for line '{}': {}",
                    line, response
                );
                so.reset()?;
                writeln!(so, "{line}")?;
                break; // Break from retry loop
            } else {
                debug!(
                    "Retrying bad response from model for line '{}' (attempt {}): {}",
                    line,
                    retry + 1,
                    response
                );
                continue; // Continue to next retry iteration
            }
        }
    }
    Ok(())
}

fn colorgrad_to_term(c: colorgrad::Color) -> termcolor::Color {
    let [r, g, b, _] = c.to_rgba8();
    termcolor::Color::Rgb(r, g, b)
}
