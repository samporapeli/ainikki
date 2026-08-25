"""Tests for TTS template rendering."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from agent.tts_template import TtsTemplate
from agent.schema import NewsItem, Source, SourceType


class TestTtsTemplate:
    """Test TTS template loading and rendering."""

    def test_loads_template_from_file(self):
        """Test that template loads from YAML file."""
        template = TtsTemplate(Path("config/tts-templates/ainikki-oletus.yaml"))
        assert template.template_path is not None
        assert "overview_heading" in template.template_data
        assert "overview_text" in template.template_data
        assert "news_item_bridges" in template.template_data

    def test_template_has_news_bridges(self):
        """Test that template has 10 items in news_item_bridges."""
        template = TtsTemplate(Path("config/tts-templates/ainikki-oletus.yaml"))
        bridges = template.template_data.get("news_item_bridges", [])
        assert len(bridges) == 10

    def test_render_basic_overview(self):
        """Test rendering with basic overview data."""
        template = TtsTemplate(Path("config/tts-templates/ainikki-oletus.yaml"))
        result = template.render({
            "overview_heading": "Test heading",
            "overview_text": "Test overview text",
            "items": []
        })
        assert "Test heading" in result
        assert "Test overview text" in result

    def test_render_with_items(self):
        """Test rendering with news items."""
        template = TtsTemplate(Path("config/tts-templates/ainikki-oletus.yaml"))

        news_items = [
            NewsItem(
                headline="OpenAI julkaisee uuden GPT-mallin",
                summary="Uusi malli on nopeampi ja tarkempa.",
                sources=[],
                rank=1,
                tags=[],
                selection_reason="test"
            ),
            NewsItem(
                headline="MikroGPT:n suosio kasvaa",
                summary="Lukijoiden rakastama yksinkertainen malli jatkaa suosioaan.",
                sources=[],
                rank=2,
                tags=[],
                selection_reason="test"
            )
        ]

        result = template.render({
            "overview_heading": "Tekoälyn kehitys",
            "overview_text": "Tänään katsomme LLM-mallien uusia edistyksiä.",
            "items": news_items
        })

        first_bridges = template.template_data["news_item_bridges"][0]
        last_bridges = template.template_data["news_item_bridges"][-1]
        assert "Tekoälyn kehitys" in result
        assert "Tänään katsomme LLM-mallien uusia edistyksiä" in result
        assert any(b in result for b in first_bridges)
        assert "Open AI julkaisee uuden GPT-mallin. Uusi malli on nopeampi ja tarkempa." in result
        assert any(b in result for b in last_bridges)
        assert "MikroGPT:n suosio kasvaa. Lukijoiden rakastama yksinkertainen malli jatkaa suosioaan." in result

    def test_pronunciation_overrides(self):
        """Test phonetic replacements for words like OpenAI and Qwen."""
        template = TtsTemplate(Path("config/tts-templates/ainikki-oletus.yaml"))
        news_items = [
            NewsItem(
                headline="Qwen 2.5 ja OpenAI julkaisevat malleja",
                summary="Alibaban Qwen kilpailee suoraan OpenAI:n kanssa.",
                sources=[],
                rank=1,
                tags=[],
                selection_reason="test",
            )
        ]
        result = template.render({
            "overview_heading": "Test",
            "overview_text": "Overview with Qwen",
            "items": news_items,
        })
        assert "Kwen" in result
        assert "Open AI" in result
        assert "Qwen" not in result

    def test_random_bridge_selection(self):
        """Test that bridges are randomly selected from options."""
        template = TtsTemplate(Path("config/tts-templates/ainikki-oletus.yaml"))

        news_items = [
            NewsItem(
                headline="Test headline",
                summary="Test summary",
                sources=[],
                rank=1,
                tags=[],
                selection_reason="test"
            ),
            NewsItem(
                headline="Test headline 2",
                summary="Test summary 2",
                sources=[],
                rank=2,
                tags=[],
                selection_reason="test"
            )
        ]

        result = template.render({
            "overview_heading": "Test",
            "overview_text": "Test overview",
            "items": news_items
        })

        assert "Test headline" in result
        assert "Test headline 2" in result

    def test_fallback_for_many_items(self):
        """Test fallback for items beyond the template range."""
        template = TtsTemplate(Path("config/tts-templates/ainikki-oletus.yaml"))

        news_items = []
        for i in range(15):
            news_items.append(NewsItem(
                headline=f"Test headline {i}",
                summary=f"Test summary {i}",
                sources=[],
                rank=i + 1,
                tags=[],
                selection_reason="test"
            ))

        result = template.render({
            "overview_heading": "Test",
            "overview_text": "Test overview",
            "items": news_items
        })

        # Beyond range items should use second-to-last bridge group, and last item should use last bridge group
        penultimate_bridges = template.template_data["news_item_bridges"][-2]
        last_bridges = template.template_data["news_item_bridges"][-1]
        assert any(b in result for b in penultimate_bridges)
        assert any(b in result for b in last_bridges)
        assert "Test headline 14. Test summary 14" in result

    def test_fallback_for_single_item(self):
        """Test fallback logic for single item."""
        template = TtsTemplate(Path("config/tts-templates/ainikki-oletus.yaml"))

        news_items = [
            NewsItem(
                headline="Test headline",
                summary="Test summary",
                sources=[],
                rank=1,
                tags=[],
                selection_reason="test"
            )
        ]

        result = template.render({
            "overview_heading": "Test",
            "overview_text": "Test overview",
            "items": news_items
        })

        # Single item is also the last item, so it should use one of the choices from the last bridge group
        last_bridges = template.template_data["news_item_bridges"][-1]
        assert any(b in result for b in last_bridges)
        assert "Test headline. Test summary" in result

    def test_missing_template_path(self):
        """Test that template works without file path."""
        template = TtsTemplate(None)
        # Should fall back to just overview text
        result = template.render({
            "overview_heading": "Test",
            "overview_text": "Test overview",
            "items": []
        })
        assert "Test overview" in result

    def test_empty_items(self):
        """Test rendering with no items."""
        template = TtsTemplate(Path("config/tts-templates/ainikki-oletus.yaml"))
        result = template.render({
            "overview_heading": "Test",
            "overview_text": "Test overview",
            "items": []
        })
        assert "Test" in result
        assert "Test overview" in result

    def test_preview_includes_all_parts(self):
        """Test that rendered output includes overview, bridges, and items."""
        template = TtsTemplate(Path("config/tts-templates/ainikki-oletus.yaml"))

        news_items = [
            NewsItem(
                headline="Test",
                summary="Test summary",
                sources=[],
                rank=1,
                tags=[],
                selection_reason="test"
            )
        ]

        result = template.render({
            "overview_heading": "Test Heading",
            "overview_text": "Test overview text describing the news",
            "items": news_items
        })

        assert "Test Heading" in result
        assert "Test overview text" in result
        assert len(result.splitlines()) >= 2  # At least overview and item

    def test_render_with_sources(self):
        """Test that source_domain and source_info are properly included."""
        template = TtsTemplate(Path("config/tts-templates/ainikki-oletus.yaml"))

        news_items = [
            NewsItem(
                headline="Ylen uutinen",
                summary="Tämä on tiivistelmä.",
                sources=[
                    Source(
                        url="https://www.yle.fi/uutiset/12345",
                        title="Alkuperäinen juttu",
                        source_type=SourceType.rss,
                    )
                ],
                rank=1,
                tags=[],
                selection_reason="test",
            )
        ]

        result = template.render({
            "overview_heading": "Test",
            "overview_text": "Test overview",
            "items": news_items,
        })

        assert "Jutun lähde: yle.fi." in result