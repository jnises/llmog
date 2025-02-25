// use std::fs::File;
// use std::io::{BufReader, Write};
// use candle_transformers::quantized_var_builder::VarBuilder;
// use hf_hub::api::sync::Api;
// use hf_hub::{Repo, RepoType};
// use tokenizers::Tokenizer;

// use candle_core::quantized::gguf_file;
// use candle_core::{Device, Tensor};
// use candle_transformers::generation::{LogitsProcessor, Sampling};

// //use candle_examples::token_output_stream::TokenOutputStream;
// use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;

// // #[derive(Parser, Debug)]
// // #[command(author, version, about, long_about = None)]
// // struct Args {
// //     /// GGUF file to load, typically a .gguf file generated by the quantize command from llama.cpp
// //     #[arg(long)]
// //     model: Option<String>,

// //     /// The initial prompt, use 'interactive' for entering multiple prompts in an interactive way
// //     /// and 'chat' for an interactive model where history of previous prompts and generated tokens
// //     /// is preserved.
// //     #[arg(long)]
// //     prompt: Option<String>,

// //     /// The length of the sample to generate (in tokens).
// //     #[arg(short = 'n', long, default_value_t = 1000)]
// //     sample_len: usize,

// //     /// The tokenizer config in json format.
// //     #[arg(long)]
// //     tokenizer: Option<String>,

// //     /// The temperature used to generate samples, use 0 for greedy sampling.
// //     #[arg(long, default_value_t = 0.8)]
// //     temperature: f64,

// //     /// Nucleus sampling probability cutoff.
// //     #[arg(long)]
// //     top_p: Option<f64>,

// //     /// Only sample among the top K samples.
// //     #[arg(long)]
// //     top_k: Option<usize>,

// //     /// The seed to use when generating random samples.
// //     #[arg(long, default_value_t = 299792458)]
// //     seed: u64,

// //     /// Enable tracing (generates a trace-timestamp.json file).
// //     #[arg(long)]
// //     tracing: bool,

// //     /// Process prompt elements separately.
// //     #[arg(long)]
// //     split_prompt: bool,

// //     /// Run on CPU rather than GPU even if a GPU is available.
// //     #[arg(long)]
// //     cpu: bool,

// //     /// Penalty to be applied for repeating tokens, 1. means no penalty.
// //     #[arg(long, default_value_t = 1.1)]
// //     repeat_penalty: f32,

// //     /// The context size to consider for the repeat penalty.
// //     #[arg(long, default_value_t = 64)]
// //     repeat_last_n: usize,

// //     /// The model size to use.
// //     #[arg(long, default_value = "phi-3b")]
// //     which: Which,

// //     #[arg(long)]
// //     use_flash_attn: bool,
// // }

// fn main() -> anyhow::Result<()> {
//     println!(
//         "avx: {}, neon: {}, simd128: {}, f16c: {}",
//         candle_core::utils::with_avx(),
//         candle_core::utils::with_neon(),
//         candle_core::utils::with_simd128(),
//         candle_core::utils::with_f16c()
//     );

//     let api = Api::new()?;
//     const MODEL_PATH: &str = "microsoft/Phi-3-mini-4k-instruct-gguf";
//     const MODEL_REVISION: &str = "999f761fe19e26cf1a339a5ec5f9f201301cbb83";
//     let repo = Repo::with_revision(MODEL_PATH.to_string(), RepoType::Model, MODEL_REVISION.to_string());
//     let api = api.repo(repo);

//     // let mut tokenizer =
//     //     Tokenizer::from_file(api.get("tokenizer.json")?).map_err(|e| anyhow::anyhow!(e))?;
//     // tokenizer.with_padding(Some(PaddingParams::default()));
//     // .with_truncation(Some(tokenizers::TruncationParams {
//     //     max_length: MAX_LENGTH,
//     //     ..Default::default()
//     // }))
//     //.map_err(|e| anyhow::anyhow!(e))?;
//     // let tokenizer = tokenizer;

//     // let mut config: Config =
//     //     serde_json::from_reader(std::fs::File::open(api.get("config.json")?)?)?;

//     // TODO: use gpu if that proves to be faster
//     // let device = if candle_core::utils::metal_is_available() {
//     //     println!("using metal");
//     //     Device::new_metal(0)?
//     // } else {
//     //     Device::cuda_if_available(0)?
//     // };
//     let device = Device::Cpu;

//     // let vb = unsafe {
//     //     VarBuilder::from_mmaped_safetensors(&[api.get("model.safetensors")?], DTYPE, &device)?
//     // };

//     // let model_path = args.model()?;
//     // let mut file = std::fs::File::open(&model_path)?;
//     // let device = candle_examples::device(args.cpu)?;
//     let model_path = api.get("Phi-3-mini-4k-instruct-q4.gguf")?;
//     let mut file = BufReader::new(File::open(&model_path)?);

//     let mut model = {
//         let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
//         // let mut total_size_in_bytes = 0;
//         // for (_, tensor) in model.tensor_infos.iter() {
//         //     let elem_count = tensor.shape.elem_count();
//         //     total_size_in_bytes +=
//         //         elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
//         // }
//         // println!(
//         //     "loaded {:?} tensors ({}) in {:.2}s",
//         //     model.tensor_infos.len(),
//         //     &format_size(total_size_in_bytes),
//         //     start.elapsed().as_secs_f32(),
//         // );
//         const USE_FLASH_ATTENTION: bool = true;
//         Phi3::from_gguf(USE_FLASH_ATTENTION, model, &mut file, &device)?
//     };
//     println!("model built");

//     let tokenizer = args.tokenizer()?;
//     let mut tos = TokenOutputStream::new(tokenizer);
//     let prompt_str = args.prompt.unwrap_or_else(|| DEFAULT_PROMPT.to_string());
//     print!("{}", &prompt_str);
//     let tokens = tos
//         .tokenizer()
//         .encode(prompt_str, true)
//         .map_err(anyhow::Error::msg)?;
//     let tokens = tokens.get_ids();
//     let to_sample = args.sample_len.saturating_sub(1);
//     let mut all_tokens = vec![];
//     let mut logits_processor = {
//         let temperature = args.temperature;
//         let sampling = if temperature <= 0. {
//             Sampling::ArgMax
//         } else {
//             match (args.top_k, args.top_p) {
//                 (None, None) => Sampling::All { temperature },
//                 (Some(k), None) => Sampling::TopK { k, temperature },
//                 (None, Some(p)) => Sampling::TopP { p, temperature },
//                 (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
//             }
//         };
//         LogitsProcessor::from_sampling(args.seed, sampling)
//     };

//     let start_prompt_processing = std::time::Instant::now();
//     let mut next_token = if !args.split_prompt {
//         let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
//         let logits = model.forward(&input, 0)?;
//         let logits = logits.squeeze(0)?;
//         logits_processor.sample(&logits)?
//     } else {
//         let mut next_token = 0;
//         for (pos, token) in tokens.iter().enumerate() {
//             let input = Tensor::new(&[*token], &device)?.unsqueeze(0)?;
//             let logits = model.forward(&input, pos)?;
//             let logits = logits.squeeze(0)?;
//             next_token = logits_processor.sample(&logits)?
//         }
//         next_token
//     };
//     let prompt_dt = start_prompt_processing.elapsed();
//     all_tokens.push(next_token);
//     if let Some(t) = tos.next_token(next_token)? {
//         print!("{t}");
//         std::io::stdout().flush()?;
//     }
//     let eos_token = *tos
//         .tokenizer()
//         .get_vocab(true)
//         .get("<|endoftext|>")
//         .unwrap();
//     let start_post_prompt = std::time::Instant::now();
//     let mut sampled = 0;
//     for index in 0..to_sample {
//         let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
//         let logits = model.forward(&input, tokens.len() + index)?;
//         let logits = logits.squeeze(0)?;
//         let logits = if args.repeat_penalty == 1. {
//             logits
//         } else {
//             let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
//             candle_transformers::utils::apply_repeat_penalty(
//                 &logits,
//                 args.repeat_penalty,
//                 &all_tokens[start_at..],
//             )?
//         };
//         next_token = logits_processor.sample(&logits)?;
//         all_tokens.push(next_token);
//         if let Some(t) = tos.next_token(next_token)? {
//             print!("{t}");
//             std::io::stdout().flush()?;
//         }
//         sampled += 1;
//         if next_token == eos_token {
//             break;
//         };
//     }
//     if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
//         print!("{rest}");
//     }
//     std::io::stdout().flush()?;
//     let dt = start_post_prompt.elapsed();
//     println!(
//         "\n\n{:4} prompt tokens processed: {:.2} token/s",
//         tokens.len(),
//         tokens.len() as f64 / prompt_dt.as_secs_f64(),
//     );
//     println!(
//         "{sampled:4} tokens generated: {:.2} token/s",
//         sampled as f64 / dt.as_secs_f64(),
//     );
//     Ok(())
// }

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

const LOG_SCORE_FORMAT: &str = "
{
    \"type\": \"object\",
    \"properties\": {
      \"reasoning\": {
        \"type\": \"string\"
      },
      \"interest_score\": {
        \"type\": \"number\"
      }
    },
    \"required\": [
      \"reasoning\",
      \"interest_score\"
    ]
  }
";

const SYSTEM_PROMPT: &str = "You are a developer log analyzer. Analyze each given line for interesting information. Use any preceding lines strictly as context. First, provide a brief reasoning; then on a new line, output 'Score: <score>', where <score> is a number from 0 to 100 based on the following scale:
- 0-20: routine/unimportant logs
- 21-40: minor information
- 41-60: noteworthy information
- 61-80: important warnings/errors
- 81-100: critical errors/security issues";

fn main() -> anyhow::Result<()> {
    const URL: &str = "http://localhost:11434";
    let agent = Agent::new_with_defaults();
    let mut pull_response = agent
        .post(format!("{URL}/api/pull"))
        .send_json(PullParams {
            //model: "llama3.2:1b-instruct-q2_K".to_string(),
            model: "llama3.2".to_string(),
            stream: false,
        })?;
    dbg!(&pull_response);
    dbg!(pull_response.body_mut().read_to_string());
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
    //for lines in windowed(reader.lines().map(|l| l.unwrap()), LINE_WINDOW) {
    for line in reader.lines() {
        let line = line?;
        //let line = lines.back().unwrap();
        if line.trim().is_empty() {
            writeln!(so, "")?;
            continue;
        }
        // let mut concatlines = String::new();
        // for (i, l) in lines.iter().enumerate() {
        //     writeln!(&mut concatlines, "{}: {l}", if i < lines.len() - 1 { "context" } else { "focus" })?;
        // }
        //let concatlines = Vec::from(lines.clone()).join("\n");
        //dbg!(&concatlines);
        // write!(
        //     so,
        //     "concatlines: {concatlines}\nendofconcatlines"
        // )?;

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
                    model: "llama3.2".to_string(),
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
                writeln!(so, "{line}")?;
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

/// Iterates over all windows of size n or less over an iterator. Only do <n for the initial items, not the last ones.
// TODO: don't clone so much
fn windowed<I: Iterator>(mut iter: I, n: usize) -> impl Iterator<Item = VecDeque<I::Item>>
where
    //I: Clone,
    I::Item: Clone,
{
    let mut current: VecDeque<I::Item> = VecDeque::new();
    std::iter::from_fn(move || {
        if let Some(item) = iter.next() {
            if current.len() >= n {
                current.pop_front();
            }
            current.push_back(item);
            Some(current.clone())
        } else {
            None
        }
    })
}
