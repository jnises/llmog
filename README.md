# llmog - LLM Log Analyzer

An experimental tool that uses large language models to automatically highlight log files based on the significance of each entry.

![screenshot](docs/screenshot.webp)

The tool processes each log entry through a pre-trained language model, which rates its importance on a scale from 0 to 100. This rating is then used to generate a color-coded output, making it easier to identify critical log entries at a glance.

Very much a prototype. The tool now supports various GGUF models runnable by `mistral.rs`. The default system prompt is designed for models capable of analytical tasks. Performance depends on the chosen model and your hardware.

If you give it something other than a log-file the model has a tendency to get quite confused. This tool uses `mistral.rs` for local model inference.

## Prerequisites

- [Rust toolchain](https://rustup.rs/)
- An internet connection is required for the first run with a new model to download it from Hugging Face Hub.

## Installation

```bash
cargo install --path .
```

Or install directly using:
```bash
cargo install --git https://github.com/jnises/llmog
```

## Usage

Pipe your logs into llmog:

```bash
# Example: Using Mistral 7B Instruct GGUF
tail -f your.log | llmog --model-repo-id "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" --model-filename "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

### Options

- `--analyze`: Show analysis after each log line
- `--model-repo-id <REPO_ID>`: Specifies the Hugging Face Hub repository ID for the model (e.g., 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF'). Mandatory.
- `--model-filename <FILENAME>`: Specifies the GGUF model filename within the Hugging Face repository (e.g., 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'). Mandatory.
- `--tokenizer-filename <FILENAME>`: Optional. Specifies the tokenizer filename (e.g., 'tokenizer.json') within the repository if it's not automatically inferred by `mistral.rs` or if you need to specify a non-standard one.
- `--gguf-log-level <LEVEL>`: Optional. Sets the log level for the GGUF loader in `mistral.rs` (e.g., 'trace', 'debug', 'info', 'warn', 'error', 'silence'). Default: 'silence'.
- `--model-cache-dir <PATH>`: Optional. Specifies a custom directory for storing downloaded Hugging Face models. Defaults to the standard Hugging Face cache directory.
- `--context=N`: Number of lines to use for context (default: 3)
- `--timeout=SECONDS`: Request timeout in seconds (default: 10). This applies to individual model inference requests. Initial model download may take longer depending on model size and internet speed.

Note that you need a terminal with true color ansi support for this to be useful.

## TODO

- Finetune using non-log files to make sure the model doesn't get confused.
