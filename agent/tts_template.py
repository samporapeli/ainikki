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

    def render(self, data: dict[str, Any]) -> str:
        """Render template with provided data.

        Data format expected:
        {
            "overview_heading": str,
            "overview_text": str,
            "items": list[NewsItem],  # where NewsItem = {headline, summary, sources, rank, tags, selection_reason}
            "closing_statement": str | None,  # override from pipeline/persona config; uses yaml pool as fallback
        }

        Template structure:
        {
            "overview_heading": "{overview_heading}",
            "overview_text": "{overview_text}",
            "bridges": list[str],
            "news_item_bridges": dict[int, list[str]],  # keyed by position (1-indexed)
            "generic_bridge_over_eight": list[str],     # fallback for items beyond position 8
            "last_item_bridge": list[str],               # bridge for the final item in the digest
            "news_items_template": str,                  # format string like "{headline}. {summary}"
            "closing_statements": list[str],             # pool for random selection fallback
        }

        Returns: fully rendered text ready for TTS.
        """
        if not self.template_data:
            return data["overview_text"]

        result_parts = []

        # Overview section
        overview_heading = self._process_field("overview_heading", data)
        overview_text = self._process_field("overview_text", data)

        if overview_heading and data.get("overview_heading"):
            result_parts.append(overview_heading)
        if overview_text:
            result_parts.append(overview_text)

        # Bridges
        bridges = self.template_data.get("bridges", [])
        bridges_processed = [self._process_field(b, data) for b in bridges]
        all_empty = all(not t for t in bridges_processed)
        if all_empty and bridges:
            result_parts.append(str(bridges[random.randrange(len(bridges))]))
        elif any(bridges_processed):
            result_parts.append("\n".join(t for t in bridges_processed if t))

        # News items
        template = self.template_data.get("news_items_template", "{headline}. {summary}")
        items = data.get("items", [])
        if items:
            news_item_bridges = self.template_data.get("news_item_bridges", {})

            for i, item in enumerate(items):
                # Determine which bridge text to use
                position = i + 1  # 1-based position
                last_idx = len(items) - 1

                if i == last_idx:
                    # Last item of the digest: use last_item_bridge
                    bridge_options = self.template_data.get("last_item_bridge", [])
                    if isinstance(bridge_options, (list, tuple)) and bridge_options:
                        bridge_text = random.choice(bridge_options)
                    else:
                        bridge_text = str(bridge_options) if bridge_options else "Seuraavaksi"
                elif position in news_item_bridges:
                    # Position within template range
                    bridge_options = news_item_bridges[position]
                    if isinstance(bridge_options, (list, tuple)) and bridge_options:
                        bridge_text = random.choice(bridge_options)
                    else:
                        bridge_text = str(bridge_options)
                else:
                    # Items beyond template range (position > 8): use generic_bridge_over_eight
                    generic = self.template_data.get("generic_bridge_over_eight", [])
                    if isinstance(generic, (list, tuple)) and generic:
                        bridge_text = random.choice(generic)
                    else:
                        bridge_text = "Jatketaan seuraavaan uutiseen"

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
                formatted = template.format(**item_dict)
                result_parts.append(f"{bridge_text}: {formatted}")

        # Closing statement: data override first, yaml pool as fallback
        closing_data = data.get("closing_statement")
        if closing_data:
            result_parts.append(closing_data)
        else:
            closing_pool = self.template_data.get("closing_statements", [])
            if closing_pool and isinstance(closing_pool, list):
                # If it's nested (list of lists), flatten first then pick
                flat_pool = [choice for opt in closing_pool for choice in (opt if isinstance(opt, list) else [opt] if opt else [])]
                if flat_pool:
                    result_parts.append(random.choice(flat_pool))

        rendered = "\n\n".join(result_parts)
        return self._apply_pronunciations(rendered)

    def _apply_pronunciations(self, text: str) -> str:
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
