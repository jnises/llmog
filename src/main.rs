use ansi_stripper::AnsiStripReader;
use clap::Parser;
use log::error;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Write as _;
use std::io::{BufRead, BufReader};
use termcolor::{ColorChoice, ColorSpec, WriteColor as _};
use ureq::Agent;

mod ansi_stripper;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Show analysis after each line
    #[arg(long)]
    analysis: bool,
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

const SYSTEM_PROMPT: &str =
    "You are a developer log analyzer. Rate each log line by uniqueness, impact, and actionability.
For each log, output EXACTLY in this format:
```
[Very brief single-sentence analysis]
SCORE: [0-100]
```
Score guide:
Low (0-30): Routine/minor info
Medium (31-70): Noteworthy/important
High (71-100): Critical/security issues";

const MODEL: &str = "llama3.2";

const LINE_WINDOW: usize = 3;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    const URL: &str = "http://localhost:11434";
    let agent = Agent::new_with_defaults();
    agent
        .post(format!("{URL}/api/pull"))
        .send_json(PullParams {
            model: MODEL.to_string(),
            stream: false,
        })?;
    let reader = BufReader::new(AnsiStripReader::new(std::io::stdin().lock()));

    let response_re =
        regex::Regex::new(r"(?i)(?<reason>.*)\n\s*(?:Score:\s*)?(?<score>\d+(?:\.\d+)?)")?;

    let stdout = termcolor::StandardStream::stdout(ColorChoice::Auto);
    let mut so = stdout.lock();
    let mut history: VecDeque<(String, String)> = VecDeque::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            writeln!(so, "")?;
            continue;
        }

        for retry in 0.. {
            let mut messages = Vec::with_capacity(history.len() + 2);
            messages.push(Message {
                role: "system".to_string(),
                content: SYSTEM_PROMPT.to_string(),
            });
            for (request, response) in &history {
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
            //dbg!(messages.iter().map(|Message{role, content}| (role, content)).collect::<Vec<_>>());
            let response: ChatResponse = agent
                .post(format!("{URL}/api/chat"))
                .send_json(ChatParams {
                    model: MODEL.to_string(),
                    stream: false,
                    messages,
                    format: None, //Some(LOG_SCORE_FORMAT.to_string()),
                })?
                .body_mut()
                .read_json()?;
            //dbg!(&response.message.content);
            if &response.message.role != "assistant" {
                anyhow::bail!("bad reponse role");
            }

            if let Some((reason, score)) =
                response_re
                    .captures(&response.message.content)
                    .and_then(|caps| {
                        Some((
                            caps["reason"].to_string(),
                            caps["score"].parse::<f64>().ok()?,
                        ))
                    })
            {
                //dbg!(&reason);
                //dbg!(score);
                // if let Some(score) = response.message.content.lines().last().and_then(|s| {
                //     //s.parse::<f64>().ok()
                //     score_re
                //         .captures(s)
                //         .and_then(|caps| caps.get(1))
                //         .and_then(|m| m.as_str().parse::<f64>().ok())
                // }) {
                let color = colorous::VIRIDIS.eval_continuous(score.clamp(0.0, 100.0) / 100.0);
                let mut color_spec = ColorSpec::new();
                color_spec.set_fg(Some(colorous_to_term(color)));
                so.set_color(&color_spec)?;
                //writeln!(so, "{reason}")?;
                write!(so, "{line}")?;
                if cli.analysis {
                    so.reset()?;
                    write!(so, " : {reason}")?;
                }
                writeln!(so, "")?;
                if history.len() > LINE_WINDOW {
                    history.pop_front();
                }
                history.push_back((line, response.message.content));
            } else {
                if retry > 10 {
                    error!("Bad response from model: {}", response.message.content);
                    so.reset()?;
                    break;
                } else {
                    continue;
                }
            }
            break;
        }
    }
    so.flush()?;
    Ok(())
}

fn colorous_to_term(c: colorous::Color) -> termcolor::Color {
    let (r, g, b) = c.as_tuple();
    termcolor::Color::Rgb(r, g, b)
}
