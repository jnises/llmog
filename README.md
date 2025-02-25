# llmog - LLM Log Analyzer

A real-time log analysis tool that uses Large Language Models to evaluate the importance of log entries.

## Features

- Real-time log analysis using local LLM (via Ollama)
- Color-coded output based on log entry importance
- Optional detailed analysis display
- Maintains context window for better log interpretation
- ANSI color code stripping for raw log input

## Prerequisites

- Rust toolchain
- [Ollama](https://ollama.ai/) installed and running locally

## Installation

```bash
cargo install --path .
```

## Usage

Pipe your logs into llmog:

```bash
tail -f your.log | llmog
```
