# llmog - LLM Log Analyzer

A real-time log analysis tool that uses Large Language Models to evaluate the importance of log entries.

Very much a prototype. Using Llama 3.2 3B currently, which is small, but may still be very slow depending on hardware.

## Features

- Real-time log analysis using local LLM (via Ollama)
- Color-coded output based on log entry importance
- Optional detailed analysis display

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

## Todo

- Handle context size properly
- Make context window configurable
