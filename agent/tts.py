"""
Text-to-speech using configurable providers.

Non-critical pipeline step — failures produce warnings, not crashes.
Returns MP3 bytes ready for Telegram upload or web embedding.
"""

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import yaml

from .schema import Briefing
from .tts_template import TtsTemplate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TtsResult:
    audio_bytes: bytes
    duration_ms: float
    raw_data: dict | None = None


TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"


def load_tts_config(config_dir: Path) -> dict[str, Any]:
    """Load TTS provider configuration from config/tts_providers.yaml."""
    config_path = config_dir / "tts_providers.yaml"
    if not config_path.exists():
        return {}

    return yaml.safe_load(config_path.read_text())


def synthesize(
    segments: list[str],
    provider_name: str,
    voice_name: str,
    max_input_bytes: int = 5000,
    client: httpx.Client | None = None,
) -> TtsResult | None:
    """Synthesize text segments into MP3 audio.

    Each segment is synthesized separately and the resulting MP3 blobs are
    concatenated. Returns TtsResult on success or None if all segments fail.
    """
    tts_config = load_tts_config(Path("config"))

    provider_cfg = tts_config.get("providers", {}).get(provider_name)
    if not provider_cfg:
        logger.error(
            "TTS skipped — provider '%s' not configured in config/tts_providers.yaml",
            provider_name,
        )
        return None

    if not provider_cfg.get("enabled", False):
        logger.error("TTS skipped — provider '%s' is disabled", provider_name)
        return None

    voice_cfg = provider_cfg.get("voices", {}).get(voice_name)
    if not voice_cfg:
        logger.error(
            "TTS skipped — voice '%s' not found for provider '%s'",
            voice_name,
            provider_name,
        )
        return None

    token_env = provider_cfg.get("token_env_var")
    if not token_env or not (token := os.environ.get(token_env)):
        logger.error("TTS skipped — provider '%s' token environment variable '%s' not set", provider_name, token_env)
        return None

    silence_path = Path("data/tts_silence_300ms.mp3")
    if not silence_path.exists():
        raise FileNotFoundError(f"TTS silence file required: {silence_path.resolve()}")
    silence = silence_path.read_bytes()

    all_audio: list[bytes] = []
    total_duration_ms = 0.0
    n_success = 0
    n_skipped = 0

    for idx, text in enumerate(segments):
        input_bytes = len(text.encode("utf-8"))
        if input_bytes > max_input_bytes:
            logger.warning(
                "TTS skipped — segment %d input %d bytes exceeds limit of %d for provider %s",
                idx, input_bytes, max_input_bytes, provider_name,
            )
            n_skipped += 1
            continue

        payload = {
            "input": {"text": text},
            "voice": {
                "name": voice_cfg["name"],
                "languageCode": voice_cfg["language"],
            },
            "audioConfig": {
                "audioEncoding": voice_cfg["audio_encoding"],
                "speakingRate": voice_cfg.get("speaking_rate", 1.0),
            },
        }

        endpoint = provider_cfg.get("endpoint", TTS_ENDPOINT)

        try:
            if client is not None:
                url = f"{endpoint}?key={token}"
                resp = client.post(url, json=payload)
            else:
                with httpx.Client(timeout=90.0, follow_redirects=True) as c:
                    url = f"{endpoint}?key={token}"
                    resp = c.post(url, json=payload)

            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as e:
            logger.error("TTS skipped — segment %d timed out: %s", idx, e)
            n_skipped += 1
            continue
        except httpx.HTTPError as e:
            logger.error("TTS skipped — segment %d HTTP error: %s", idx, e)
            n_skipped += 1
            continue

        audio_content = data.get("audioContent")
        if not audio_content:
            logger.error("TTS skipped — segment %d API returned no audioContent", idx)
            n_skipped += 1
            continue

        all_audio.append(_decode_audio(audio_content))
        all_audio.append(silence)
        total_duration_ms += data.get("audioConfig", {}).get("sampleRateHertz", 0)
        n_success += 1

    # Strip trailing silence
    if len(all_audio) > 1:
        all_audio.pop()

    if not all_audio:
        logger.warning("TTS skipped — no segments synthesized (0/%d)", len(segments))
        return None

    raw_data = {"provider": provider_name, "voice": voice_name, "n_segments": len(segments), "n_success": n_success, "n_skipped": n_skipped}

    return TtsResult(
        audio_bytes=b"".join(all_audio),
        duration_ms=total_duration_ms,
        raw_data=raw_data,
    )


def _decode_audio(content: str) -> bytes:
    """Decode base64 audio content from various encoding prefixes."""
    start = content.find(";base64,")
    if start != -1:
        raw = content[start + 8:]
    elif content.startswith("data:"):
        raw = content.split(",", 1)[1]
    else:
        raw = content
    return base64.b64decode(raw)


DEFAULT_MAX_INPUT_BYTES = 5000


def render_script(briefing: Briefing, config_dir: Path = Path("config")) -> tuple[list[str], str, str]:
    """Render the narration text for a Briefing using its topic configuration.

    Returns:
        (segments, provider_name, voice_name) — segments is a list of strings,
        each suitable for a single TTS API request.
    """
    topic_config_path = config_dir / "topics" / f"{briefing.topic}.yaml"
    topic_cfg = yaml.safe_load(topic_config_path.read_text()) if topic_config_path.exists() else {}
    tts_cfg = topic_cfg.get("tts", {})

    provider = tts_cfg.get("provider", "google-cloud")
    voice = tts_cfg.get("voice", "fi-FI-Chirp3-HD-Callirrhoe")
    template_rel = tts_cfg.get("template_path", "tts-templates/ainikki-oletus.yaml")
    template_path = config_dir / template_rel

    template = TtsTemplate(template_path)
    segments = template.render({
        "overview_heading": briefing.digest_topic,
        "overview_text": briefing.overview,
        "items": briefing.items,
    })
    return segments, provider, voice


def test_cli():
    import argparse
    import json
    import sys
    from .schema import Briefing

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="TTS CLI — Text-to-Speech")
    parser.add_argument("--digest", help="Path to digest JSON file (e.g. data/output/ai_daily_2026-08-25.json)")
    parser.add_argument("--text", help="Plain text to synthesize directly")
    parser.add_argument("--render-only", "--dry-run", action="store_true", help="Only render the narration text and display length stats without calling TTS API")
    parser.add_argument("--output", "-o", help="Output file path (default: auto for --digest, /tmp/tts_sample.mp3 for --text)")
    parser.add_argument("--provider", default=None, help="Provider name override from config/tts_providers.yaml")
    parser.add_argument("--voice", default=None, help="Voice name override from config/tts_providers.yaml")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_INPUT_BYTES, help="Maximum allowed input bytes limit (default: 5000)")
    parser.add_argument("--debug", "-d", action="store_true", help="Print full API response before writing")
    args = parser.parse_args()

    max_bytes = args.max_bytes

    if args.digest:
        digest_path = Path(args.digest)
        if not digest_path.exists():
            sys.exit(f"Digest file not found: {digest_path}")

        briefing = Briefing(**json.loads(digest_path.read_text()))
        segments, default_provider, default_voice = render_script(briefing)

        provider = args.provider or default_provider
        voice = args.voice or default_voice
        default_out = digest_path.with_name(f"{digest_path.stem}_audio.mp3")
        output_path = Path(args.output) if args.output else default_out
    elif args.text:
        segments = [args.text]
        provider = args.provider or "google-cloud"
        voice = args.voice or "fi-FI-Chirp3-HD-Callirrhoe"
        output_path = Path(args.output) if args.output else Path("/tmp/tts_sample.mp3")
    else:
        sys.exit("No input given. Use --digest <path> or --text <string>")

    total_chars = sum(len(s) for s in segments)
    total_bytes = sum(len(s.encode("utf-8")) for s in segments)
    print(f"--- Script Preview ({len(segments)} segments, {total_chars} chars, {total_bytes}/{max_bytes} bytes) ---")
    for idx, segment in enumerate(segments):
        seg_bytes = len(segment.encode("utf-8"))
        flag = " OVER LIMIT" if seg_bytes > max_bytes else ""
        print(f"  [{idx}] {seg_bytes} bytes{flag}:")
        print(segment[:200])
        print()
    print("-" * 70)

    if args.render_only:
        return

    for seg in segments:
        if len(seg.encode("utf-8")) > max_bytes:
            sys.exit(f"Error: segment {segments.index(seg)} ({len(seg.encode('utf-8'))} bytes) exceeds limit ({max_bytes} bytes).")

    result = synthesize(segments, provider, voice, max_input_bytes=max_bytes)
    if result is None:
        sys.exit("TTS synthesis returned no result (check GOOGLE_CLOUD_API_TOKEN or provider configuration).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result.audio_bytes)
    print(f"Audio saved to {output_path} ({len(result.audio_bytes)} bytes, ~{result.duration_ms / 1000:.1f}s)")


if __name__ == "__main__":
    test_cli()
