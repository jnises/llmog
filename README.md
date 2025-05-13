# llmog - LLM Log Analyzer

An experimental tool that uses large language models to automatically highlight log files based on the significance of each entry.

![screenshot](docs/screenshot.webp)

The tool processes each log entry through a pre-trained language model, which rates its importance on a scale from 0 to 100. This rating is then used to generate a color-coded output, making it easier to identify critical log entries at a glance.

Very much a prototype. Using a finetuned Gemma 3 1B, which is small, but may still be very slow depending on hardware.

If you give it something other than a log-file the model has a tendency to get quite confused.

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

Note that you need a terminal with true color ansi support for this to be useful.

## Todo

- Handle context size properly
