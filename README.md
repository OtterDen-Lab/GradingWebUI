# GradingWebUI

A web-based interface for grading exams with Canvas LMS integration and AI-assisted workflows.

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Set up Canvas API credentials

Create a `.env` file in your working directory:

```bash
CANVAS_API_KEY=your_canvas_api_key_here
CANVAS_API_URL=https://your-institution.instructure.com
```

### 2. Run the server

```bash
python -m grading_web_ui.web_api.main
```

### 3. Docker quick start (optional)

From the repo root:

```bash
cd docker/web-grading
docker compose up --build
```

Then open:

```
http://localhost:8765
```

For more details, see `docker/web-grading/README.md`.

## Features

### Web UI Capabilities

- **Problem-first grading**: Grade all Q1, then Q2, etc. with intelligent ordering
- **AI assistance**: Name extraction, blank detection, handwriting transcription
- **Canvas integration**: Dev/prod environment switching with safe defaults
- **Persistent sessions**: Resume grading anytime with database-backed state
- **Duplicate detection**: SHA256 hashing prevents re-processing the same files
- **Local storage**: SQLite database for FERPA-friendly workflows

## Configuration

The web UI reads Canvas credentials from `~/.env` by default. To use production Canvas, add:

```bash
USE_PROD_CANVAS=true
CANVAS_API_KEY_PROD=your_prod_key
CANVAS_API_URL_PROD=https://your-institution.instructure.com
```

## Requirements

- Python >= 3.12
- Docker (optional, for containerized runs)
- Canvas API access
- Optional: OpenAI/Anthropic/Ollama for AI-powered features

## Documentation

For detailed documentation, see `WebUI/docs`.

## License

This project is licensed under the GPL-3.0-or-later license. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/OtterDen-Lab/Autograder).
