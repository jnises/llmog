# llmog - LLM Log Analyzer

An experimental tool that uses large language models to automatically highlight log files based on the significance of each entry.

![screenshot](docs/screenshot.webp)

The tool processes each log entry through a pre-trained language model, which rates its importance on a scale from 0 to 100. This rating is then used to generate a color-coded output, making it easier to identify critical log entries at a glance.

llmog now uses the [Candle](https://github.com/huggingface/candle) ML framework for local inference, replacing the previous Ollama backend. This means you no longer need a separate Ollama server running.

The default model is [`jnises/gemma-3-1b-llmog-GGUF`](https://huggingface.co/jnises/gemma-3-1b-llmog-GGUF). Model files (GGUF format for the model itself, and a tokenizer configuration) are downloaded automatically from the Hugging Face Hub on the first run via the `hf-hub` library.

Very much a prototype. Using a finetuned Gemma 3 1B, which is small, but may still be very slow depending on hardware, especially if running on CPU.

If you give it something other than a log-file the model has a tendency to get quite confused.

## Prerequisites

- [Rust toolchain](https://rustup.rs/)
- For GPU acceleration (recommended):
    - NVIDIA GPU: CUDA toolkit installed. Candle will attempt to use CUDA if available.
    - Other GPUs: Support may vary. Check Candle documentation for details on Metal (macOS) or Vulkan.

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
tail -f your.log | llmog
```

### Options

- `--analyze`: Show analysis after each log line.
- `--context=N`: Number of lines to use for context (default: 3).
- `--max-tokens=N`: Maximum number of tokens to generate for the analysis (default: 200, formerly `--timeout`).
- `--cpu`: Force CPU usage, even if a GPU (e.g., CUDA-enabled) is available.
- `--model-repo <REPO_ID>`: Specify a different Hugging Face model repository (e.g., `another-user/another-model-GGUF`). Defaults to `jnises/gemma-3-1b-llmog-GGUF`.
- `--gguf-file <FILENAME>`: Specify a different GGUF filename from the repository (e.g., `model-Q4_K_M.gguf`). Defaults to `model-Q8_0.gguf`.
- `--tokenizer-file <FILENAME>`: Specify a different tokenizer filename from the repository (e.g., `tokenizer.json`). Defaults to `tokenizer.json`.
- `--seed <SEED>`: Seed for random number generation during sampling (default: 299792458).
- `--temperature <TEMP>`: Temperature for sampling. Higher values (e.g., 0.7) make output more random, lower values (e.g., 0.3) make it more deterministic.
- `--top-p <PROB>`: Top-P probability for nucleus sampling. Filters vocabulary to the smallest set of tokens whose cumulative probability exceeds P.
- `--repeat-penalty <PENALTY>`: Penalty for repeating tokens (default: 1.1). Values > 1 discourage repetition.
- `--repeat-last-n <N>`: Context window size for repeat penalty (default: 64).

### Removed Options

- `--ollama-url`: No longer used as Ollama is not the backend.

Note that you need a terminal with true color ANSI support for this to be useful.

## TODO

- Finetune using non-log files to make sure the model doesn't get confused.
- Explore performance improvements for CPU inference.
- Investigate support for more quantized model formats or backends via Candle.
