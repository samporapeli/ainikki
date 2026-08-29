"""
Template rendering for TTS narration.
Supports YAML-based templates with nested lists for random phrase selection.
"""

import re
import random
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class TtsTemplate:
    """Template for TTS narration. Loads from YAML and renders with provided data."""

    def __init__(self, template_path: Path | None = None):
        self.template_path = template_path
        self.template_data: dict[str, Any] = {}

        if template_path and template_path.exists():
            self.template_data = self._load_template(template_path)

    def _load_template(self, template_path: Path) -> dict[str, Any]:
        """Load template configuration from YAML file."""
        import yaml
        return yaml.safe_load(template_path.read_text())

    def render(self, data: dict[str, Any]) -> list[str]:
        """Render template into TTS segments, one per audio chunk.

        Data format expected:
        {
            "overview_heading": str,
            "overview_text": str,
            "items": list[NewsItem],
            "closing_statement": str | None,
        }

        Template structure:
        {
            "overview_heading": "{overview_heading}",
            "overview_text": "{overview_text}",
            "bridges": list[str],
            "news_item_bridges": dict[int, list[str]],
            "generic_bridge_over_eight": list[str],
            "last_item_bridge": list[str],
            "news_items_template": str,
            "closing_statements": list[str],
        }

        Returns: list of segment strings, each fitting in one TTS request.
        """
        if not self.template_data:
            return [data["overview_text"]]

        segments: list[str] = []

        # Overview section (segment 0)
        overview_parts = []
        overview_heading = self._process_field("overview_heading", data)
        overview_text = self._process_field("overview_text", data)
        if overview_heading and data.get("overview_heading"):
            overview_parts.append(overview_heading)
        if overview_text:
            overview_parts.append(overview_text)
        if overview_parts:
            segments.append(self._pronounce("\n\n".join(overview_parts)))

        # Bridge from overview to first news item (appended to segment 0
        # when short; if overview is already large, it may overflow — that's
        # fine, TTS limit catches it per segment).
        bridges = self.template_data.get("bridges", [])
        bridges_processed = [self._process_field(b, data) for b in bridges]
        all_empty = all(not t for t in bridges_processed)
        bridge_text = ""
        if all_empty and bridges:
            bridge_text = str(bridges[random.randrange(len(bridges))])
        elif any(bridges_processed):
            bridge_text = "\n".join(t for t in bridges_processed if t)
        if bridge_text and segments:
            segments[0] = segments[0] + "\n\n" + bridge_text
        elif bridge_text:
            segments.append(self._pronounce(bridge_text))

        # News items (one segment per item)
        nitem_tpl = self.template_data.get("news_items_template", "{headline}. {summary}")
        items = data.get("items", [])
        if items:
            news_item_bridges = self.template_data.get("news_item_bridges", {})

            for i, item in enumerate(items):
                position = i + 1
                last_idx = len(items) - 1

                if i == last_idx:
                    bridge_options = self.template_data.get("last_item_bridge", [])
                    if isinstance(bridge_options, (list, tuple)) and bridge_options:
                        bridge = random.choice(bridge_options)
                    else:
                        bridge = str(bridge_options) if bridge_options else "Seuraavaksi"
                elif position in news_item_bridges:
                    bridge_options = news_item_bridges[position]
                    if isinstance(bridge_options, (list, tuple)) and bridge_options:
                        bridge = random.choice(bridge_options)
                    else:
                        bridge = str(bridge_options)
                else:
                    generic = self.template_data.get("generic_bridge_over_eight", [])
                    if isinstance(generic, (list, tuple)) and generic:
                        bridge = random.choice(generic)
                    else:
                        bridge = "Jatketaan seuraavaan uutiseen"

                source_domain = ""
                if getattr(item, "sources", None):
                    domain = urlparse(str(item.sources[0].url)).netloc
                    source_domain = domain[4:] if domain.startswith("www.") else domain

                source_info_tpl = self.template_data.get("source_info_template", " Jutun lähde: {source_domain}.")
                source_info = source_info_tpl.format(source_domain=source_domain) if source_domain else ""

                item_dict = {
                    **{k: str(v) for k, v in item.model_dump().items()},
                    "source_domain": source_domain,
                    "source_info": source_info,
                }
                formatted = nitem_tpl.format(**item_dict)
                segments.append(self._pronounce(f"{bridge}: {formatted}"))

        # Closing statement (final segment)
        closing_data = data.get("closing_statement")
        if closing_data:
            segments.append(self._pronounce(closing_data))
        else:
            closing_pool = self.template_data.get("closing_statements", [])
            if closing_pool and isinstance(closing_pool, list):
                flat_pool = [choice for opt in closing_pool for choice in (opt if isinstance(opt, list) else [opt] if opt else [])]
                if flat_pool:
                    segments.append(self._pronounce(random.choice(flat_pool)))

        return segments

    def _pronounce(self, text: str) -> str:
        """Apply phonetic pronunciation overrides configured in template."""
        pronunciations = self.template_data.get("pronunciations")
        if not pronunciations or not isinstance(pronunciations, dict):
            return text

        for term, phonetic in pronunciations.items():
            if term and phonetic:
                text = re.sub(rf"\b{re.escape(str(term))}\b", str(phonetic), text)
        return text

    def _process_field(self, field_name: str, data: dict[str, Any]) -> str:
        """Process a field from template data."""
        template_value = self.template_data.get(field_name)
        if not template_value:
            return ""

        if isinstance(template_value, str):
            # Standard string placeholder replacement
            data_value = data.get(field_name)
            if data_value is not None:
                return str(template_value).format(**{field_name: data_value})
            return template_value
        return str(template_value)

    def _process_list(self, list_value: list, data: dict[str, Any]) -> str:
        """Process a list field from template data."""
        if not isinstance(list_value, list):
            return str(list_value)

        list_parts = []
        for item in list_value:
            if isinstance(item, str):
                processed_item = self._process_field(item, data)
                list_parts.append(processed_item)
            else:
                list_parts.append(str(item))

        return "\n".join(list_parts)
