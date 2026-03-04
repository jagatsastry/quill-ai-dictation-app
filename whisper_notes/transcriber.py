from pathlib import Path
import whisper


class TranscriptionError(RuntimeError):
    pass


class Transcriber:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None  # lazy load on first use

    def _load_model(self):
        if self._model is None:
            self._model = whisper.load_model(self.model_name)

    def transcribe(self, audio_path: Path) -> str:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        self._load_model()
        try:
            result = self._model.transcribe(str(audio_path))
            return result["text"].strip()
        except Exception as e:
            raise TranscriptionError(str(e)) from e
