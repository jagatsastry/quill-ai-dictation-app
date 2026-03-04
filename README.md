# whisper-notes

macOS menu bar app for voice notetaking. Records locally, transcribes with [OpenAI Whisper](https://github.com/openai/whisper), summarizes with [Ollama](https://ollama.ai), saves to `~/Notes/` as markdown.

## Requirements

- macOS 13+
- Python 3.11+
- [uv](https://github.com/astral-sh/uv): `pip install uv`
- [ffmpeg](https://ffmpeg.org): `brew install ffmpeg`
- [Ollama](https://ollama.ai) running locally with a model pulled (default: `gemma2:9b`)

## Install

```bash
git clone https://github.com/YOUR_USERNAME/whisper-notes
cd whisper-notes
uv venv && source .venv/bin/activate
uv sync
```

## Run

```bash
whisper-notes
# or
python -m whisper_notes.app
```

The app appears in your macOS menu bar as 🎙 Whisper Notes.

## Usage

1. Click **🎙 Whisper Notes** in the menu bar
2. Click **Start Recording** — the icon changes to ⏺ Recording...
3. Speak your note
4. Click **Stop Recording** — Whisper transcribes, Ollama summarizes
5. A notification appears when the note is saved to `~/Notes/`

If Ollama is offline, the raw transcript is saved without a summary.

## Configure

Set environment variables before running:

| Env var | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Whisper model: tiny / base / small / medium / large |
| `OLLAMA_MODEL` | `gemma2:9b` | Any model in `ollama list` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_TIMEOUT` | `60` | Seconds before Ollama request times out |
| `NOTES_DIR` | `~/Notes` | Where notes are saved |

## Test

```bash
uv sync --group dev
pytest -v
```

## Note format

Each note saved as `~/Notes/YYYY-MM-DD-HH-MM.md`:

```markdown
# Note — 2026-03-04 14:32

## Summary
- Key point one
- Key point two

## Transcript
Full raw transcript here...

---
*Recorded: 2026-03-04 14:32:17 | Duration: 1m 23s | Model: base*
```
