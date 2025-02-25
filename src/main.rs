use ansi_stripper::AnsiStripReader;
use itertools::Itertools as _;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::io::Write as _;
use std::io::{BufRead, BufReader};
use termcolor::{ColorChoice, ColorSpec, WriteColor as _};
use ureq::{Agent, http::StatusCode};

mod ansi_stripper;

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

const SYSTEM_PROMPT: &str = "You are a developer log analyzer. Rate how interesting each given line is from a developers's perspective. Rate redundant information lower. First, provide a very brief reasoning; then on a new line, output 'Score: <score>', where <score> is a number from 0 to 100 based on the following scale:
- 0-20: routine/unimportant logs
- 21-40: minor information
- 41-60: noteworthy information
- 61-80: important warnings/errors
- 81-100: critical errors/security issues";

fn main() -> anyhow::Result<()> {
    const URL: &str = "http://localhost:11434";
    let agent = Agent::new_with_defaults();
    const MODEL: &str = "llama3.2";
    //const MODEL: &str = "llama3.2:1b";
    let mut pull_response = agent
        .post(format!("{URL}/api/pull"))
        .send_json(PullParams {
            model: MODEL.to_string(),
            stream: false,
        })?;
    dbg!(&pull_response);
    dbg!(pull_response.body_mut().read_to_string()?);
    let reader = BufReader::new(AnsiStripReader::new(std::io::stdin().lock()));

    //     const SYSTEM_PROMPT: &str = "You are a developer log analyzer. Given a sequence of log lines rate how interesting the LAST line is from a developer's perspective. Start with a brief reasoning. Then on a new line, rate the interestingness from 0 to 100, where:
    //     - 0-20: routine/unimportant logs
    //     - 21-40: minor information
    //     - 41-60: noteworthy information
    //     - 61-80: important warnings/errors
    //     - 81-100: critical errors/security issues
    //     Output format must be exactly:
    //     <reason>
    //     Score: <score>";
    // const SYSTEM_PROMPT: &str = "You are a developer log analyzer. Given a sequence of log lines rate how interesting the last line is from a developer's perspective. Rate the interestingness from 0 to 100, where:
    // - 0-20: routine/unimportant logs
    // - 21-40: minor information
    // - 41-60: noteworthy information
    // - 61-80: important warnings/errors
    // - 81-100: critical errors/security issues
    // Output format must be exactly:
    // Score: <score>";

    // the model sometimes likes to prepend Score: even though we tell it not to
    //let score_re = regex::Regex::new(r"\s*(?i)(?:Score:\s*)?(\d+(?:\.\d+)?)")?;

    let response_re =
        regex::Regex::new(r"(?i)(?<reason>.*)\n\s*(?:Score:\s*)?(?<score>\d+(?:\.\d+)?)")?;

    let stdout = termcolor::StandardStream::stdout(ColorChoice::Auto);
    let mut so = stdout.lock();
    const LINE_WINDOW: usize = 3;
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
                let color = colorous::TURBO.eval_continuous(score.clamp(0.0, 100.0) / 100.0);
                let mut color_spec = ColorSpec::new();
                color_spec.set_fg(Some(colorous_to_term(color)));
                so.set_color(&color_spec)?;
                //writeln!(so, "{reason}")?;
                write!(so, "{line}")?;
                const SHOW_REASON: bool = true;
                if SHOW_REASON {
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
                    writeln!(
                        std::io::stderr(),
                        "bad response from model: {}",
                        response.message.content
                    )?;
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
