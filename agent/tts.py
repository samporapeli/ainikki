"""
Text-to-speech using configurable providers.

Non-critical pipeline step — failures produce warnings, not crashes.
Generates Ogg Opus audio via Google Cloud TTS, with ffmpeg handling
silence insertion and segment concatenation.
"""

import base64
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from .schema import Briefing
from .tts_template import TtsTemplate

logger = logging.getLogger(__name__)


TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"

# Google Cloud TTS defaults for OGG_OPUS
_DEFAULT_OGG_SAMPLE_RATE = 24000  # Hz (Google Cloud default)
_DEFAULT_OGG_BITRATE = "64k"
_DEFAULT_SILENCE_MS = 600  # gap between segments
DEFAULT_MAX_INPUT_BYTES = 5000


@dataclass(frozen=True)
class TtsResult:
    """Synthesized audio result."""
    audio_bytes: bytes
    duration_ms: float
    raw_data: dict | None = None


def load_tts_config(config_dir: Path) -> dict[str, Any]:
    """Load TTS provider configuration from config/tts_providers.yaml."""
    config_path = config_dir / "tts_providers.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text())


def _find_ffmpeg() -> str | None:
    """Find ffmpeg on PATH. Returns path string or None."""
    return shutil.which("ffmpeg")


def _ffmpeg_available(ffmpeg_path: str) -> bool:
    """Check that ffmpeg is usable (not broken or empty)."""
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _decode_audio_content(content: str) -> bytes:
    """Decode base64 audio content from TTS API response.
    
    Handles data URI prefixes (data:;base64, ...) or raw base64.
    """
    start = content.find(";base64,")
    if start != -1:
        raw = content[start + 8:]
    elif content.startswith("data:"):
        raw = content.split(",", 1)[1]
    else:
        raw = content
    return base64.b64decode(raw)


# Backward-compat alias for tests
_decode_audio = _decode_audio_content


def _synthesize_segment_ogg(
    text: str,
    voice_cfg: dict[str, Any],
    provider_cfg: dict[str, Any],
    token: str,
    client: httpx.Client | None = None,
) -> bytes | None:
    """Synthesize one text segment via Google Cloud TTS.
    
    Returns OGG_OPUS bytes, or None on failure.
    """
    payload = {
        "input": {"text": text},
        "voice": {
            "name": voice_cfg["name"],
            "languageCode": voice_cfg["language"],
        },
        "audioConfig": {
            "audioEncoding": "OGG_OPUS",
            "speakingRate": voice_cfg.get("speaking_rate", 1.0),
        },
    }
    endpoint = provider_cfg.get("endpoint", TTS_ENDPOINT)

    try:
        if client is not None:
            resp = client.post(f"{endpoint}?key={token}", json=payload)
        else:
            with httpx.Client(timeout=90.0, follow_redirects=True) as c:
                resp = c.post(f"{endpoint}?key={token}", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException as e:
        logger.error("TTS segment timed out: %s", e)
        return None
    except httpx.HTTPError as e:
        logger.error("TTS segment HTTP error: %s", e)
        return None

    audio_content = data.get("audioContent")
    if not audio_content:
        logger.warning("TTS returned no audioContent")
        return None

    return _decode_audio_content(audio_content)


def _generate_ogg_silence(
    duration_ms: int,
    ffmpeg_path: str,
    sample_rate: int = _DEFAULT_OGG_SAMPLE_RATE,
) -> bytes | None:
    """Generate silence as OGG_OPUS via ffmpeg.
    
    Uses anullsrc to produce silence, encodes as libopus in an Ogg container.
    Returns OGG_OPUS bytes, or None on failure.
    """
    cmd = [
        ffmpeg_path, "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", f"{duration_ms / 1000:.3f}",
        "-c:a", "libopus",
        "-b:a", _DEFAULT_OGG_BITRATE,
        "-f", "ogg",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=10)
    if result.returncode != 0:
        logger.error(
            "ffmpeg silence generation failed (status %d): %s",
            result.returncode, result.stderr.decode(errors="replace")[:200],
        )
        return None
    if not result.stdout:
        logger.warning("ffmpeg produced empty silence output")
        return None
    return result.stdout


def _ffmpeg_concat_multi(
    inputs: list[Path],
    ffmpeg_path: str,
) -> bytes | None:
    """Concatenate multiple OGG_OPUS files using ffmpeg's audio concat filter.
    
    This is more robust than the concat demuxer for OGG files because it
    decodes/re-encodes and handles serialisation differences.
    
    Returns OGG_OPUS bytes (read from stdout), or None on failure.
    """
    if not inputs:
        return None
    
    if len(inputs) == 1:
        data = inputs[0].read_bytes()
        if data[:4] != b"OggS":
            logger.warning("single file missing OggS header — not valid Ogg")
            return None
        return data

    n = len(inputs)
    cmd = [ffmpeg_path, "-y"]
    for inp_path in inputs:
        cmd.extend(["-i", str(inp_path)])
    cmd.extend([
        "-filter_complex", f"concat=n={n}:v=0:a=1",
        "-c:a", "libopus",
        "-b:a", _DEFAULT_OGG_BITRATE,
        "-f", "ogg",
        "pipe:1",
    ])

    result = subprocess.run(
        cmd, capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        stderr_str = result.stderr.decode(errors="replace")
        logger.error(
            "ffmpeg concat failed (status %d): %s",
            result.returncode, stderr_str[:500],
        )
        return None
    if not result.stdout:
        logger.warning("ffmpeg concat produced empty output")
        return None
    if result.stdout[:4] != b"OggS":
        logger.warning("ffmpeg concat output missing OggS header — not valid Ogg")
        return None
    return result.stdout



def synthesize(
    segments: list[str],
    provider_name: str,
    voice_name: str,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    client: httpx.Client | None = None,
    ffmpeg_path: str | None = None,
) -> TtsResult | None:
    """Synthesize text segments into Ogg Opus audio.
    
    Each segment is synthesized separately via Google Cloud TTS.
    A silence gap is inserted between consecutive segments (generated via ffmpeg).
    All parts are concatenated via ffmpeg using the concat demuxer
    (file list piped via stdin, no temp file needed).
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
            voice_name, provider_name,
        )
        return None

    token_env = provider_cfg.get("token_env_var")
    if not token_env or not (token := os.environ.get(token_env)):
        logger.error(
            "TTS skipped — env var '%s' not set for provider '%s'",
            token_env, provider_name,
        )
        return None

    # Find ffmpeg
    ffmpeg = ffmpeg_path or _find_ffmpeg()
    if not ffmpeg:
        logger.error("TTS skipped — ffmpeg not found in PATH")
        return None
    if not _ffmpeg_available(ffmpeg):
        logger.error("TTS skipped — ffmpeg not usable")
        return None

    silence_ms = int(voice_cfg.get("silence_between_segments_ms", _DEFAULT_SILENCE_MS))

    # Synthesize each segment via Google Cloud TTS
    seg_bytes: list[bytes] = []
    n_success = 0
    n_skipped = 0

    for idx, text in enumerate(segments):
        if len(text.encode("utf-8")) > max_input_bytes:
            logger.warning(
                "TTS skipped — segment %d (%d bytes) exceeds %d byte limit for provider %s",
                idx, len(text.encode("utf-8")), max_input_bytes, provider_name,
            )
            n_skipped += 1
            continue

        seg = _synthesize_segment_ogg(text, voice_cfg, provider_cfg, token, client)
        if seg is None:
            logger.warning("TTS failed for segment %d", idx)
            n_skipped += 1
            continue
        seg_bytes.append(seg)
        n_success += 1

    if n_success == 0:
        logger.warning("TTS synthesized 0/%d segments", len(segments))
        return None

    # Concatenate via ffmpeg (interleave segments + silences)
    audio_bytes = _synthesize_with_silence(
        seg_bytes, silence_ms, ffmpeg,
    )
    if audio_bytes is None:
        logger.error("TTS synthesis failed — concatenation with silence failed")
        return None

    approx_duration = round(
        n_success * 3.0 + (n_success - 1) * silence_ms / 1000.0,
        2,  # rough estimate; exact needs OGG frame parsing
    )

    raw_data = {
        "provider": provider_name,
        "voice": voice_name,
        "codec": "opus",
        "container": "ogg",
        "n_segments": len(segments),
        "n_success": n_success,
        "n_skipped": n_skipped,
        "silence_between_ms": silence_ms,
    }

    return TtsResult(
        audio_bytes=audio_bytes,
        duration_ms=approx_duration,
        raw_data=raw_data,
    )


def _synthesize_with_silence(
    seg_bytes: list[bytes],
    silence_ms: int,
    ffmpeg_path: str,
) -> bytes | None:
    """Write segments + silences to a temp dir, then concatenate via ffmpeg.
    
    Uses a temp directory to hold intermediate files; cleaned up on exit.
    """
    with tempfile.TemporaryDirectory(prefix="ainikki_ogg_") as tmpdir:
        tmp = Path(tmpdir)

        # Write valid segments (those starting with OggS) to temp files
        seg_paths: list[Path] = []
        for i, b in enumerate(seg_bytes):
            if b[:4] != b"OggS":
                logger.warning("segment %d has no OggS header — not a valid Ogg", i)
                continue
            (tmp / f"seg_{i:04d}.ogg").write_bytes(b)
            seg_paths.append(tmp / f"seg_{i:04d}.ogg")

        # Generate silences between consecutive segments
        sil_paths: list[Path] = []
        for i in range(len(seg_bytes) - 1):
            sil = _generate_ogg_silence(silence_ms, ffmpeg_path)
            if sil is None:
                logger.warning("Failed to generate silence %d, skipping", i)
                continue
            if sil[:4] != b"OggS":
                logger.warning("silence %d has no OggS header — not a valid Ogg", i)
                continue
            (tmp / f"sil_{i:04d}.ogg").write_bytes(sil)
            sil_paths.append(tmp / f"sil_{i:04d}.ogg")

        # Interleave: seg0, sil0, seg1, sil1, ..., segN
        concat_paths: list[Path] = []
        for i, p in enumerate(seg_paths):
            concat_paths.append(p)
            if i < len(sil_paths):
                concat_paths.append(sil_paths[i])

        return _ffmpeg_concat_multi(concat_paths, ffmpeg_path)


def render_script(briefing: Briefing, config_dir: Path = Path("config")) -> tuple[list[str], str, str]:
    """Render the narration text for a Briefing using its topic configuration.

    Returns:
        (segments, provider_name, voice_name)
    """
    topic_config_path = config_dir / "topics" / f"{briefing.topic}.yaml"
    topic_cfg = yaml.safe_load(topic_config_path.read_text()) if topic_config_path.exists() else {}
    tts_cfg = topic_cfg.get("tts", {})

    provider = tts_cfg.get("provider", "google-cloud")
    voice = tts_cfg.get("voice", "fi-FI-Chirp3-HD-Achird")
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

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="TTS CLI — Text-to-Speech (Ogg Opus)",
    )
    parser.add_argument("--digest", help="Path to digest JSON file")
    parser.add_argument("--text", help="Plain text to synthesize")
    parser.add_argument("--render-only", "-n", action="store_true",
                        help="Only render the narration text")
    parser.add_argument("--output", "-o", help="Output file path (.ogg)")
    parser.add_argument("--provider", default=None, help="Provider override")
    parser.add_argument("--voice", default=None, help="Voice override")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_INPUT_BYTES,
                        help="Input byte limit")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Print full API response")
    args = parser.parse_args()

    if args.digest:
        digest_path = Path(args.digest)
        if not digest_path.exists():
            sys.exit(f"digest file not found: {digest_path}")

        briefing = Briefing(**json.loads(digest_path.read_text()))
        segments, default_provider, default_voice = render_script(briefing)
        provider = args.provider or default_provider
        voice = args.voice or default_voice
        default_out = digest_path.with_name(
            f"{digest_path.stem}_audio.ogg"
        )
        output_path = Path(args.output) if args.output else default_out
    elif args.text:
        segments = [args.text]
        provider = args.provider or "google-cloud"
        voice = args.voice or "fi-FI-Chirp3-HD-Achird"
        output_path = Path(args.output) if args.output else Path("/tmp/tts_sample.ogg")
    else:
        sys.exit("No input given. Use --digest <path> or --text <string>")

    total_chars = sum(len(s) for s in segments)
    total_bytes = sum(len(s.encode("utf-8")) for s in segments)
    print(f"--- [{len(segments)} segs, {total_chars} chars, {total_bytes}/{args.max_bytes} bytes] ---")
    for i, s in enumerate(segments):
        b = len(s.encode("utf-8"))
        flag = " OVER LIMIT" if b > args.max_bytes else ""
        print(f"  [{i}] {b} bytes{flag}:")
        print(s[:200])
        print()
    print("-" * 70)

    if args.render_only:
        return

    for seg in segments:
        if len(seg.encode("utf-8")) > args.max_bytes:
            sys.exit(
                f"segment {segments.index(seg)} "
                f"({len(seg.encode('utf-8'))} bytes) exceeds limit "
                f"({args.max_bytes} bytes)."
            )

    result = synthesize(segments, provider, voice, max_input_bytes=args.max_bytes)
    if result is None:
        sys.exit("TTS returned no result (check config/API key/ffmpeg).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result.audio_bytes)
    print(f"Saved to {output_path} ({len(result.audio_bytes)} bytes, "
          f"~{result.duration_ms / 1000:.1f}s)")


if __name__ == "__main__":
    test_cli()
