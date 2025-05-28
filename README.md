# llmog - LLM Log Analyzer

An experimental tool that uses large language models to automatically highlight log files based on the significance of each entry.

![screenshot](docs/screenshot.webp)

The tool processes each log entry through a pre-trained language model, which rates its importance on a scale from 0 to 100. This rating is then used to generate a color-coded output, making it easier to identify critical log entries at a glance.

This is very much a prototype. It uses a finetuned Gemma 3 1B model, which is small but may still be very slow depending on hardware.

The model is served using Hugging Face Hub.

## Prerequisites

- [Rust toolchain](https://rustup.rs/)
- [Ollama](https://ollama.ai/) installed and running locally

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

- `--analyze`: Show analysis after each log line
- `--ollama-url=URL`: Set custom Ollama API URL (default: http://localhost:11434)
- `--context=N`: Number of lines to use for context (default: 3)
- `--timeout=SECONDS`: Request timeout in seconds (default: 10)

Note that you need a terminal with true color ANSI support for this to be useful.

## Training

Notebooks for finetuning the language model are provided in the `training` directory.
The notebooks are meant to be run on Google Colab.

They need the secrets `HF_TOKEN` and `OPENROUTER_API_KEY`.

In order to upload new data or models to Hugging Face, make sure you have things set up there. Replace `jnises` with your username in the notebooks.

### Synthesizing data

The notebook in [training/synthesize.ipynb](training/synthesize.ipynb) uses the `gemini-2.0-flash` model accessed using [openrouter](https://openrouter.ai/).
A paid account there is required. Synthesizing all the data should cost $4 or so.

The notebook synthesizes log files and non-log files.
From those files, conversations are synthesized using a system prompt that mostly matches the one used later during inference.

### Finetuning

Finetuning is done using [training/finetune.ipynb](training/finetune.ipynb).
The notebook uses [unsloth](https://unsloth.ai/).

### Quantizing

Quantizing the model is done in a separate notebook [training/quantize.ipynb](training/quantize.ipynb).
