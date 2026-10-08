"""Validate a topic configuration file for the digest runner.

Used by ``run_digest.sh`` to read and validate a topic's YAML config without
re-parsing YAML inline in the shell. Prints the topic's ``period`` and its
``telegram_channel_id_env`` name, tab-separated, so the shell can read them
safely::

    daily<TAB>TELEGRAM_CHAT_ID_FOSS

Exits non-zero with a message on stderr when the config is missing or invalid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ALLOWED_PERIODS = {"daily", "weekly", "monthly"}


def load_topic_config(path: Path) -> str:
    """Return the validated ``period<TAB>telegram_channel_id_env`` string."""
    if not path.exists():
        raise SystemExit(f"Topic configuration not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    period = data.get("period")
    channel_env = data.get("telegram_channel_id_env")
    if period not in ALLOWED_PERIODS:
        raise SystemExit(f"Unsupported period for {path}: {period!r}")
    if not isinstance(channel_env, str) or not channel_env:
        raise SystemExit(f"Missing telegram_channel_id_env in {path}")
    return f"{period}\t{channel_env}"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("Usage: agent.topic_config <topic-config.yaml>")
    print(load_topic_config(Path(argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
