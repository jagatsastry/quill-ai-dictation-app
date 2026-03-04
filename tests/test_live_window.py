import sys

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_tk():
    """Mock tkinter entirely — it can't run headless."""
    tk_mock = MagicMock()
    scrolledtext_mock = MagicMock()
    with patch.dict(
        "sys.modules", {"tkinter": tk_mock, "tkinter.scrolledtext": scrolledtext_mock}
    ):
        if "whisper_notes.live_window" in sys.modules:
            del sys.modules["whisper_notes.live_window"]
        import whisper_notes.live_window as lw

        yield lw, tk_mock


def test_live_window_creates_root(mock_tk):
    lw, tk_mock = mock_tk
    on_close = MagicMock()
    win = lw.LiveWindow(on_close=on_close)
    tk_mock.Tk.assert_called_once()


def test_append_schedules_update(mock_tk):
    lw, tk_mock = mock_tk
    on_close = MagicMock()
    win = lw.LiveWindow(on_close=on_close)
    win.append("hello")
    win.root.after.assert_called()


def test_close_triggers_callback(mock_tk):
    lw, tk_mock = mock_tk
    on_close = MagicMock()
    win = lw.LiveWindow(on_close=on_close)
    win._on_close()
    on_close.assert_called_once()


def test_destroy_is_safe_to_call_twice(mock_tk):
    lw, tk_mock = mock_tk
    win = lw.LiveWindow(on_close=MagicMock())
    win.destroy()
    win.destroy()  # should not raise


def test_append_after_destroy_is_noop(mock_tk):
    lw, tk_mock = mock_tk
    win = lw.LiveWindow(on_close=MagicMock())
    win.destroy()
    win.append("hello")  # should not raise


def test_get_text_after_destroy_returns_empty(mock_tk):
    lw, tk_mock = mock_tk
    win = lw.LiveWindow(on_close=MagicMock())
    win.destroy()
    assert win.get_text() == ""
