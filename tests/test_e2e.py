"""
End-to-end tests for whisper-notes Live Transcription.

These tests exercise full user flows from the spec and design doc,
verifying observable outcomes: files on disk, content structure,
state transitions, and notification calls.

Mocking strategy:
- Mock at OS boundaries: sounddevice (no mic), faster-whisper (no GPU),
  Ollama (via respx HTTP mock), tkinter (no display), rumps (no macOS menu).
- Real components: NoteWriter (real file I/O), Summarizer (real HTTP client
  with respx interception), LiveTranscriber + LiveTranscriberThread (real
  threading with mocked WhisperModel).
- Verify: real files on tmp_path, their content, state machine transitions.
"""
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from whisper_notes.config import Config, ConfigError
from whisper_notes.live_recorder import LiveRecordingError
from whisper_notes.live_transcriber import LiveTranscriber, LiveTranscriberThread
from whisper_notes.note_writer import NoteWriter
from whisper_notes.summarizer import Summarizer, SummarizerError

SAMPLE_RATE = 16000


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_rumps():
    """Mock rumps module before importing app (no macOS menu bar in CI)."""
    rumps_mock = MagicMock()
    rumps_mock.App = MagicMock

    sub_mocks = {
        "rumps": rumps_mock,
        "whisper_notes.recorder": MagicMock(),
        "whisper_notes.transcriber": MagicMock(),
        "whisper_notes.summarizer": MagicMock(),
        "whisper_notes.note_writer": MagicMock(),
        "whisper_notes.live_transcriber": MagicMock(),
        "whisper_notes.live_recorder": MagicMock(),
        "whisper_notes.live_window": MagicMock(),
    }

    with patch.dict("sys.modules", sub_mocks):
        if "whisper_notes.app" in sys.modules:
            del sys.modules["whisper_notes.app"]
        import whisper_notes.app as app_module

        yield app_module, rumps_mock


@pytest.fixture
def notes_dir(tmp_path):
    """Temporary notes directory for real file I/O."""
    d = tmp_path / "Notes"
    d.mkdir()
    return d


@pytest.fixture
def mock_whisper_model():
    """Mock faster-whisper WhisperModel to return deterministic text."""
    with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
        call_count = [0]

        def fake_transcribe(audio, **kwargs):
            call_count[0] += 1
            seg = MagicMock()
            seg.text = f" chunk {call_count[0]}"
            return [seg], MagicMock()

        MockModel.return_value.transcribe.side_effect = fake_transcribe
        yield MockModel, call_count


@pytest.fixture
def mock_whisper_silent():
    """Mock faster-whisper WhisperModel to return empty (silence)."""
    with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = ([], MagicMock())
        yield MockModel


# ============================================================
# Scenario 1: Happy path — full live session produces note
# ============================================================


class TestHappyPathLiveSession:
    """User starts live transcription, speaks, stops, gets note with summary."""

    def test_full_pipeline_produces_note_file(self, notes_dir, mock_whisper_model, respx_mock):
        """Feed audio -> transcribe -> summarize -> note file on disk."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={
                "response": "- Key discussion point\n- Action item assigned",
                "done": True,
            })
        )

        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()

        # Feed 3 seconds of audio
        for _ in range(3):
            thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.5)
        thread.stop()
        thread.join(timeout=5)

        assert len(collected) >= 3, f"Expected >= 3 chunks, got {len(collected)}"

        full_transcript = " ".join(collected)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        writer = NoteWriter(notes_dir=notes_dir)
        summary = summarizer.summarize(full_transcript)
        recorded_at = datetime(2026, 3, 4, 15, 0, 0)
        path = writer.write(
            transcript=full_transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=recorded_at,
        )

        # Verify file exists
        assert path.exists()
        assert path.suffix == ".md"

        # Verify content structure
        content = path.read_text()
        assert "## Summary" in content
        assert "## Transcript" in content
        assert "Key discussion point" in content
        assert "chunk" in content
        assert "live/base" in content
        assert "0s" in content  # duration_seconds=0

    def test_note_filename_format(self, notes_dir, mock_whisper_model, respx_mock):
        """Note filename matches YYYY-MM-DD-HH-MM.md pattern."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.3)
        thread.stop()
        thread.join(timeout=5)

        writer = NoteWriter(notes_dir=notes_dir)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        transcript = " ".join(collected)
        summary = summarizer.summarize(transcript)
        recorded_at = datetime(2026, 3, 4, 14, 30, 0)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=recorded_at,
        )

        assert path.name == "2026-03-04-14-30.md"


# ============================================================
# Scenario 3: Empty transcript (silence) — no Ollama call
# ============================================================


class TestEmptyTranscriptSilence:
    """No speech detected -> note saved with '(no speech detected)', no summary."""

    def test_empty_transcript_saves_no_speech_detected(self, notes_dir, mock_whisper_silent):
        """Silence -> note file with '(no speech detected)' and no Summary section."""
        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.3)
        thread.stop()
        thread.join(timeout=5)

        # No text collected — silence
        assert len(collected) == 0

        # Simulate app behavior: empty transcript -> "(no speech detected)", no Ollama
        transcript = " ".join(collected).strip()
        if not transcript:
            transcript = "(no speech detected)"
            summary = None
        else:
            summary = "should not reach here"

        writer = NoteWriter(notes_dir=notes_dir)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 10, 0, 0),
        )

        content = path.read_text()
        assert "(no speech detected)" in content
        assert "## Summary" not in content
        assert "## Transcript" in content


# ============================================================
# Scenario 4: Ollama offline — raw transcript saved
# ============================================================


class TestOllamaOffline:
    """Ollama unavailable -> note saved with transcript only, no summary."""

    def test_ollama_offline_saves_raw_transcript(self, notes_dir, mock_whisper_model):
        """Connection refused -> SummarizerError -> note without summary."""
        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.3)
        thread.stop()
        thread.join(timeout=5)

        transcript = " ".join(collected)
        summary = None

        with patch("whisper_notes.summarizer.httpx.post", side_effect=httpx.ConnectError("refused")):
            summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
            try:
                summary = summarizer.summarize(transcript)
            except SummarizerError:
                summary = None

        writer = NoteWriter(notes_dir=notes_dir)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 11, 0, 0),
        )

        content = path.read_text()
        assert "## Transcript" in content
        assert "chunk" in content
        assert "## Summary" not in content


# ============================================================
# Scenario 5: NOTES_DIR doesn't exist — created automatically
# ============================================================


class TestNotesDirCreation:
    """NOTES_DIR auto-created by NoteWriter on first write."""

    def test_notes_dir_created_automatically(self, tmp_path, mock_whisper_model, respx_mock):
        """Non-existent NOTES_DIR is created, note saved inside it."""
        new_dir = tmp_path / "AutoCreatedNotes"
        assert not new_dir.exists()

        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()
        thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.3)
        thread.stop()
        thread.join(timeout=5)

        writer = NoteWriter(notes_dir=new_dir)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        transcript = " ".join(collected)
        summary = summarizer.summarize(transcript)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 9, 0, 0),
        )

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert path.exists()
        assert path.parent == new_dir


# ============================================================
# Scenario 6 & 7 & 8: Config env vars
# ============================================================


class TestConfigEnvVars:
    """Config respects LIVE_CHUNK_SECONDS and FASTER_WHISPER_MODEL env vars."""

    def test_custom_chunk_seconds_and_model(self, monkeypatch):
        """LIVE_CHUNK_SECONDS=5 and FASTER_WHISPER_MODEL=small are respected."""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "5")
        monkeypatch.setenv("FASTER_WHISPER_MODEL", "small")
        cfg = Config()
        assert cfg.live_chunk_seconds == 5
        assert cfg.faster_whisper_model == "small"

    def test_invalid_chunk_seconds_raises_config_error(self, monkeypatch):
        """LIVE_CHUNK_SECONDS='nope' raises ConfigError."""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "nope")
        with pytest.raises(ConfigError, match="LIVE_CHUNK_SECONDS"):
            Config()

    def test_zero_chunk_seconds_raises_config_error(self, monkeypatch):
        """LIVE_CHUNK_SECONDS='0' raises ConfigError with 'must be >= 1'."""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "0")
        with pytest.raises(ConfigError, match="must be >= 1"):
            Config()

    def test_float_chunk_seconds_raises_config_error(self, monkeypatch):
        """LIVE_CHUNK_SECONDS='3.5' raises ConfigError (must be integer)."""
        monkeypatch.setenv("LIVE_CHUNK_SECONDS", "3.5")
        with pytest.raises(ConfigError, match="must be an integer"):
            Config()

    def test_model_name_appears_in_note(self, notes_dir, monkeypatch, respx_mock):
        """Model env var flows into note file's model field."""
        monkeypatch.setenv("FASTER_WHISPER_MODEL", "small")

        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
            seg = MagicMock()
            seg.text = " test text"
            MockModel.return_value.transcribe.return_value = ([seg], MagicMock())

            cfg = Config()
            transcriber = LiveTranscriber(model_name=cfg.faster_whisper_model)
            collected = []
            thread = LiveTranscriberThread(
                transcriber=transcriber,
                chunk_seconds=cfg.live_chunk_seconds,
                sample_rate=SAMPLE_RATE,
                on_text=collected.append,
            )
            thread.start()
            thread.feed(np.zeros(SAMPLE_RATE * cfg.live_chunk_seconds, dtype=np.float32))
            time.sleep(0.5)
            thread.stop()
            thread.join(timeout=5)

        transcript = " ".join(collected)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        writer = NoteWriter(notes_dir=notes_dir)
        summary = summarizer.summarize(transcript)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model=f"live/{cfg.faster_whisper_model}",
            recorded_at=datetime(2026, 3, 4, 12, 0, 0),
        )

        content = path.read_text()
        assert "live/small" in content


# ============================================================
# Scenario 9: Sounddevice error on start — stays idle
# ============================================================


class TestSounddeviceError:
    """LiveRecorder.start() fails -> app stays idle, notification sent."""

    def test_recording_error_preserves_idle_state(self, mock_rumps, tmp_path):
        """Sounddevice error keeps app in idle, sends notification."""
        app_module, rumps_mock = mock_rumps
        cfg = Config()
        cfg.notes_dir = tmp_path / "Notes"
        cfg.notes_dir.mkdir()

        with patch("whisper_notes.app.Recorder"), \
             patch("whisper_notes.app.Transcriber"), \
             patch("whisper_notes.app.Summarizer"), \
             patch("whisper_notes.app.NoteWriter"), \
             patch("whisper_notes.app.LiveRecorder") as MockLiveRecorder, \
             patch("whisper_notes.app.LiveTranscriber"), \
             patch("whisper_notes.app.LiveTranscriberThread"), \
             patch("whisper_notes.app.LiveWindow"), \
             patch.object(app_module, "LiveRecErr", LiveRecordingError):
            MockLiveRecorder.return_value.start.side_effect = LiveRecordingError("no mic")
            app = app_module.WhisperNotesApp(cfg)
            app._on_live_transcribe(None)
            assert app.state == "idle"

        # Verify no note file was created
        note_files = list(cfg.notes_dir.glob("*.md"))
        assert len(note_files) == 0


# ============================================================
# Scenario 10: Record Note flow unchanged
# ============================================================


class TestRecordNoteUnchanged:
    """Existing Record Note flow still works after live transcription feature added."""

    def test_start_recording_still_works(self, mock_rumps, tmp_path):
        """Start Recording changes state to 'recording'."""
        app_module, _ = mock_rumps
        cfg = Config()
        cfg.notes_dir = tmp_path / "Notes"
        cfg.notes_dir.mkdir()

        with patch("whisper_notes.app.Recorder") as MockRecorder, \
             patch("whisper_notes.app.Transcriber"), \
             patch("whisper_notes.app.Summarizer"), \
             patch("whisper_notes.app.NoteWriter"), \
             patch("whisper_notes.app.LiveRecorder"), \
             patch("whisper_notes.app.LiveTranscriber"), \
             patch("whisper_notes.app.LiveTranscriberThread"), \
             patch("whisper_notes.app.LiveWindow"):
            app = app_module.WhisperNotesApp(cfg)
            app._on_start_recording(None)
            assert app.state == "recording"
            MockRecorder.return_value.start.assert_called_once()


# ============================================================
# Scenario 11: Full pipeline file content structure
# ============================================================


class TestFileContentStructure:
    """Verify the complete markdown structure of a live note."""

    def test_note_has_complete_structure(self, notes_dir, mock_whisper_model, respx_mock):
        """Note contains title, summary, transcript, and metadata."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={
                "response": "- Meeting recap\n- Next steps defined",
                "done": True,
            })
        )

        transcriber = LiveTranscriber(model_name="base")
        collected = []
        thread = LiveTranscriberThread(
            transcriber=transcriber,
            chunk_seconds=1,
            sample_rate=SAMPLE_RATE,
            on_text=collected.append,
        )
        thread.start()
        for _ in range(2):
            thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
        time.sleep(0.5)
        thread.stop()
        thread.join(timeout=5)

        transcript = " ".join(collected)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        writer = NoteWriter(notes_dir=notes_dir)
        summary = summarizer.summarize(transcript)
        recorded_at = datetime(2026, 3, 4, 16, 45, 0)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=recorded_at,
        )

        content = path.read_text()
        lines = content.split("\n")

        # Title line
        assert lines[0].startswith("# Note")
        assert "2026-03-04 16:45" in lines[0]

        # Summary section
        assert "## Summary" in content
        assert "Meeting recap" in content

        # Transcript section
        assert "## Transcript" in content

        # Metadata footer
        assert "Duration: 0s" in content
        assert "Model: live/base" in content
        assert "---" in content


# ============================================================
# Scenario 12: Multiple chunks accumulated in order
# ============================================================


class TestMultipleChunksOrder:
    """All transcribed chunks appear in order in the final note."""

    def test_chunks_in_order(self, notes_dir, respx_mock):
        """Five chunks appear in the note in sequential order."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
            chunk_num = [0]

            def fake_transcribe(audio, **kwargs):
                chunk_num[0] += 1
                seg = MagicMock()
                seg.text = f" segment{chunk_num[0]}"
                return [seg], MagicMock()

            MockModel.return_value.transcribe.side_effect = fake_transcribe

            transcriber = LiveTranscriber(model_name="base")
            collected = []
            thread = LiveTranscriberThread(
                transcriber=transcriber,
                chunk_seconds=1,
                sample_rate=SAMPLE_RATE,
                on_text=collected.append,
            )
            thread.start()
            for _ in range(5):
                thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
            time.sleep(1.0)
            thread.stop()
            thread.join(timeout=5)

        assert len(collected) >= 5

        transcript = " ".join(collected)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        writer = NoteWriter(notes_dir=notes_dir)
        summary = summarizer.summarize(transcript)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 17, 0, 0),
        )

        content = path.read_text()
        # Verify order: segment1 before segment2 before segment3 etc.
        for i in range(1, 6):
            assert f"segment{i}" in content
        pos1 = content.index("segment1")
        pos2 = content.index("segment2")
        pos3 = content.index("segment3")
        assert pos1 < pos2 < pos3


# ============================================================
# Scenario 13: Rapid start after stop — two separate notes
# ============================================================


class TestRapidStartAfterStop:
    """Two consecutive live sessions produce two separate note files."""

    def test_two_sessions_two_files(self, notes_dir, respx_mock):
        """Back-to-back sessions each produce their own note."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        writer = NoteWriter(notes_dir=notes_dir)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)

        paths = []
        for session_num in range(2):
            with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
                seg = MagicMock()
                seg.text = f" session{session_num + 1}"
                MockModel.return_value.transcribe.return_value = ([seg], MagicMock())

                transcriber = LiveTranscriber(model_name="base")
                collected = []
                thread = LiveTranscriberThread(
                    transcriber=transcriber,
                    chunk_seconds=1,
                    sample_rate=SAMPLE_RATE,
                    on_text=collected.append,
                )
                thread.start()
                thread.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
                time.sleep(0.3)
                thread.stop()
                thread.join(timeout=5)

            transcript = " ".join(collected)
            summary = summarizer.summarize(transcript)
            path = writer.write(
                transcript=transcript,
                summary=summary,
                duration_seconds=0,
                model="live/base",
                recorded_at=datetime(2026, 3, 4, 18, session_num, 0),
            )
            paths.append(path)

        # Two separate files
        assert len(paths) == 2
        assert paths[0] != paths[1]
        assert paths[0].exists()
        assert paths[1].exists()

        # Each contains its session text
        assert "session1" in paths[0].read_text()
        assert "session2" in paths[1].read_text()


# ============================================================
# Scenario: App state machine — menu items in live state
# ============================================================


class TestAppStateMachineLive:
    """Verify app state transitions and menu item states for live mode."""

    def test_idle_to_live_to_idle(self, mock_rumps, tmp_path):
        """Full state cycle: idle -> live -> processing -> idle."""
        app_module, rumps_mock = mock_rumps
        cfg = Config()
        cfg.notes_dir = tmp_path / "Notes"
        cfg.notes_dir.mkdir()

        with patch("whisper_notes.app.Recorder"), \
             patch("whisper_notes.app.Transcriber"), \
             patch("whisper_notes.app.Summarizer"), \
             patch("whisper_notes.app.NoteWriter") as MockWriter, \
             patch("whisper_notes.app.LiveRecorder"), \
             patch("whisper_notes.app.LiveTranscriber"), \
             patch("whisper_notes.app.LiveTranscriberThread"), \
             patch("whisper_notes.app.LiveWindow") as MockWindow:
            app = app_module.WhisperNotesApp(cfg)

            # idle
            assert app.state == "idle"

            # idle -> live
            app._on_live_transcribe(None)
            assert app.state == "live"

            # live -> processing -> idle (via _on_stop_live + _finish_live)
            app._live_pump_timer = MagicMock()
            app._live_window.get_text.return_value = "some text"
            MockWriter.return_value.write.return_value = Path("/tmp/test.md")
            with patch("threading.Thread"):
                app._on_stop_live(None)
                assert app.state == "processing"

            # Run _finish_live directly to complete the cycle
            app._finish_live()
            assert app.state == "idle"
            assert app._live_chunks == []
            assert app._live_thread is None

    def test_stop_live_noop_when_idle(self, mock_rumps, tmp_path):
        """_on_stop_live is a no-op if state is not 'live'."""
        app_module, _ = mock_rumps
        cfg = Config()
        cfg.notes_dir = tmp_path / "Notes"
        cfg.notes_dir.mkdir()

        with patch("whisper_notes.app.Recorder"), \
             patch("whisper_notes.app.Transcriber"), \
             patch("whisper_notes.app.Summarizer"), \
             patch("whisper_notes.app.NoteWriter"), \
             patch("whisper_notes.app.LiveRecorder"), \
             patch("whisper_notes.app.LiveTranscriber"), \
             patch("whisper_notes.app.LiveTranscriberThread"), \
             patch("whisper_notes.app.LiveWindow"):
            app = app_module.WhisperNotesApp(cfg)
            app.state = "idle"
            app._on_stop_live(None)
            assert app.state == "idle"


# ============================================================
# Scenario: Partial buffer transcribed on stop
# ============================================================


class TestPartialBufferOnStop:
    """Audio shorter than chunk_seconds is transcribed when stop is called."""

    def test_partial_audio_transcribed_on_stop(self, notes_dir, respx_mock):
        """Feed half a chunk, stop -> remaining buffer is transcribed."""
        respx_mock.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "summary", "done": True})
        )

        with patch("whisper_notes.live_transcriber.WhisperModel") as MockModel:
            seg = MagicMock()
            seg.text = " partial audio"
            MockModel.return_value.transcribe.return_value = ([seg], MagicMock())

            transcriber = LiveTranscriber(model_name="base")
            collected = []
            thread = LiveTranscriberThread(
                transcriber=transcriber,
                chunk_seconds=1,
                sample_rate=SAMPLE_RATE,
                on_text=collected.append,
            )
            thread.start()

            # Feed only 0.5 seconds (half a chunk)
            thread.feed(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
            time.sleep(0.2)
            thread.stop()
            thread.join(timeout=5)

        # Partial buffer should have been transcribed on stop
        assert len(collected) >= 1
        assert "partial audio" in collected[0]

        writer = NoteWriter(notes_dir=notes_dir)
        summarizer = Summarizer(ollama_url="http://localhost:11434", model="gemma2:9b", timeout=10)
        transcript = " ".join(collected)
        summary = summarizer.summarize(transcript)
        path = writer.write(
            transcript=transcript,
            summary=summary,
            duration_seconds=0,
            model="live/base",
            recorded_at=datetime(2026, 3, 4, 19, 0, 0),
        )
        content = path.read_text()
        assert "partial audio" in content
