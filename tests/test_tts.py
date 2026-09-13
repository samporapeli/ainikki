"""Tests for TTS provider configuration and synthesis."""

import base64
import httpx
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from agent.tts import synthesize, load_tts_config


class TestLoadTtsConfig:
    """Test TTS config loading from config/tts_providers.yaml."""

    def test_loads_google_cloud_config(self):
        config = load_tts_config(Path("config"))
        assert "providers" in config
        assert "google-cloud" in config["providers"]

    def test_provider_enabled_flag(self):
        config = load_tts_config(Path("config"))
        provider = config["providers"].get("google-cloud")
        assert provider is not None
        assert provider.get("enabled") is True

    def test_has_fi_fi_voice(self):
        config = load_tts_config(Path("config"))
        provider = config["providers"].get("google-cloud")
        voices = provider.get("voices", {})
        assert "fi-FI-Chirp3-HD-Achird" in voices


class TestTtsSynthesize:
    """Test TTS synthesis functionality."""

    def test_returns_none_for_missing_config(self):
        result = synthesize(["test"], "nonexistent-provider", "default-voice")
        assert result is None

    def test_returns_none_for_disabled_provider(self):
        with patch.dict(os.environ, {"GOOGLE_CLOUD_API_TOKEN": "test-token"}):
            with patch("agent.tts.load_tts_config") as mock_load:
                mock_load.return_value = {"providers": {"disabled-provider": {"enabled": False, "voices": {}}}}
                result = synthesize(["test"], "disabled-provider", "default-voice")
                assert result is None

    def test_returns_none_for_missing_token(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_API_TOKEN", raising=False)
        result = synthesize(["test"], "google-cloud", "fi-FI-Chirp3-HD-Achird")
        assert result is None

    def test_returns_none_for_too_long_input(self):
        long_text = "A" * 10000
        result = synthesize([long_text], "google-cloud", "fi-FI-Chirp3-HD-Achird", max_input_bytes=5000)
        assert result is None

    def test_synthesize_success_with_mocked_client(self):
        fake_ogg = b"OggS" + b"\x00" * 128  # fake OGG_OPUS bytes
        fake_b64 = base64.b64encode(fake_ogg).decode()
        with patch.dict(os.environ, {"GOOGLE_CLOUD_API_TOKEN": "mock-token"}):
            with patch("httpx.Client.post") as mock_post:
                mock_post.return_value.raise_for_status.return_value = None
                mock_post.return_value.json.return_value = {
                    "audioContent": fake_b64,
                    "audioConfig": {"sampleRateHertz": 24000},
                }
                with patch("agent.tts._find_ffmpeg", return_value="/fake/ffmpeg"), \
                     patch("agent.tts._ffmpeg_available", return_value=True):
                    result = synthesize(["Test text"], "google-cloud", "fi-FI-Chirp3-HD-Achird")
                    assert result is not None
                    assert result.audio_bytes == fake_ogg
                    assert result.duration_ms == 3.0
                    assert result.raw_data["provider"] == "google-cloud"
                    assert result.raw_data["voice"] == "fi-FI-Chirp3-HD-Achird"
                    assert result.raw_data["codec"] == "opus"
                    assert result.raw_data["container"] == "ogg"
                    assert result.raw_data["n_segments"] == 1
                    assert result.raw_data["n_success"] == 1
                    assert result.raw_data["n_skipped"] == 0

    def test_synthesize_missing_audiocontent(self):
        with patch.dict(os.environ, {"GOOGLE_CLOUD_API_TOKEN": "mock-token"}):
            with patch("httpx.Client.post") as mock_post:
                mock_post.return_value.raise_for_status.return_value = None
                mock_post.return_value.json.return_value = {"audioContent": None, "audioConfig": {"sampleRateHertz": 24000}}
                result = synthesize(["Test text"], "google-cloud", "fi-FI-Chirp3-HD-Achird")
                assert result is None

    def test_synthesize_returns_none_when_ffmpeg_missing(self):
        fake_ogg = b"OggS_test_bytes"
        with patch.dict(os.environ, {"GOOGLE_CLOUD_API_TOKEN": "mock-token"}):
            with patch("httpx.Client.post") as mock_post:
                mock_post.return_value.raise_for_status.return_value = None
                mock_post.return_value.json.return_value = {
                    "audioContent": base64.b64encode(fake_ogg).decode(),
                }
                with patch("agent.tts._find_ffmpeg", return_value=None):
                    result = synthesize(["Test text"], "google-cloud", "fi-FI-Chirp3-HD-Achird")
                    assert result is None

    def test_synthesize_multiple_segments_with_silence(self):
        fake_ogg = b"OggS" + b"\x00" * 100  # fake OGG_OPUS
        fake_b64 = base64.b64encode(fake_ogg).decode()
        with patch.dict(os.environ, {"GOOGLE_CLOUD_API_TOKEN": "mock-token"}):
            with patch("httpx.Client.post") as mock_post:
                mock_post.return_value.raise_for_status.return_value = None
                mock_post.return_value.json.return_value = {
                    "audioContent": fake_b64,
                }
                with patch("agent.tts._find_ffmpeg", return_value="/fake/ffmpeg"), \
                     patch("agent.tts._ffmpeg_available", return_value=True), \
                     patch("agent.tts._generate_ogg_silence", return_value=b"OggS\x01" * 50), \
                     patch("agent.tts._ffmpeg_concat_multi") as mock_concat:
                    mock_concat.return_value = b"OggS_concatenated_x" * 20
                    result = synthesize(
                        ["Seg one", "Seg two"],
                        "google-cloud", "fi-FI-Chirp3-HD-Achird",
                    )
                    assert result is not None
                    assert result.raw_data["n_segments"] == 2
                    assert result.raw_data["n_success"] == 2
                    assert result.raw_data["silence_between_ms"] == 600


class TestTtsDecodeAudio:
    """Test audio decoding from various formats."""

    def test_plain_base64(self):
        from agent.tts import _decode_audio
        encoded = "VGVzdA=="
        assert _decode_audio(encoded) == b"Test"

    def test_base64_with_prefix(self):
        from agent.tts import _decode_audio
        encoded = "data:audio/mp3;base64,VGVzdA=="
        assert _decode_audio(encoded) == b"Test"

    def test_data_uri_format(self):
        from agent.tts import _decode_audio
        encoded = "data:audio/mp3,VGVzdA=="
        assert _decode_audio(encoded) == b"Test"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
