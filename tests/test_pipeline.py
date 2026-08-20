import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from agent.pipeline import run_pipeline
from agent.score import ScoreValidationError
from agent.validate import EmptyBriefingError
from agent.collect.hn import parse_hn_hits
from agent.schema import Period, Briefing

CLEAN_HTML = Path("tests/fixtures/article_clean.html").read_text()


def _load_test_raw_items():
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    return parse_hn_hits(fixture["hits"])  # 4 items


def _make_llm_handler(score_min=3, score_max=10):
    """Single handler serving all four LLM steps (cluster/score/
    compose/overview), which may use different providers
    (OpenAI-compatible OR Anthropic, see config/models.yaml: compose
    uses anthropic/claude-sonnet-4-6 while others use OpenRouter).
    Step is identified from the system prompt content.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        is_anthropic = request.url.host == "api.anthropic.com"

        if is_anthropic:
            system_prompt = body.get("system", "")
            user_content = body["messages"][0]["content"]
        else:
            system_prompt = body["messages"][0]["content"]
            user_content = body["messages"][1]["content"]

        candidate_lines = [line for line in user_content.splitlines() if line.strip().startswith("[")]
        n = len(candidate_lines)

        if "uutisanalyytikko" in system_prompt:
            clusters = [{"candidate_indices": [i], "primary_index": 0, "reason": None} for i in range(n)]
            content = json.dumps({"clusters": clusters})
        elif "ehdokaslistaa" in system_prompt:
            k = max(score_min, min(n, score_max))
            selected = [{"candidate_index": i, "rank": i + 1, "selection_reason": f"peruste {i}"}
                        for i in range(k)]
            cutoff = score_min
            content = json.dumps({"selected": selected, "cutoff_rank": cutoff})
        elif "aiheeseen liittyvä" in system_prompt:
            content = json.dumps({"results": [{"index": i, "keep": True} for i in range(n)]})
        elif "Kohdeyleisö:" in system_prompt:
            content = json.dumps({"headline": "Testiotsikko juttu",
                                   "summary": "Testiyhteenveto joka kuvaa juttua lyhyesti."})
        elif "Tiivistelmäotsikko" in system_prompt:
            content = json.dumps({"overview": "Päivän aiheet liittyivät tekoälyyn ja sen kehitykseen.",
                                   "digest_topic": "Tekoäly ja sen kehitys"})
        else:
            raise AssertionError(f"unrecognized system prompt: {system_prompt[:100]}")

        if is_anthropic:
            return httpx.Response(200, json={"content": [{"type": "text", "text": content}]},
                                   headers={"x-request-id": "test"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.0002},
        })

    return handler


def _enrich_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=CLEAN_HTML)


def test_full_pipeline_happy_path(tmp_path):
    tmp_out = tmp_path / "out"
    tmp_data = tmp_path / "data"
    raw_items = _load_test_raw_items()
    llm_client = httpx.Client(transport=httpx.MockTransport(_make_llm_handler()))
    enrich_client = httpx.Client(transport=httpx.MockTransport(_enrich_handler))

    since = datetime(2026, 7, 16, tzinfo=timezone.utc)
    until = since + timedelta(days=1)

    path = run_pipeline(
        topic="ai", period=Period.daily, since=since, until=until,
        config_dir=Path("config"), data_dir=tmp_data, out_dir=tmp_out,
        raw_items_override=raw_items, llm_client=llm_client, enrich_client=enrich_client,
    )

    assert path.exists()
    briefing = Briefing(**json.loads(path.read_text()))

    assert briefing.topic == "ai"
    assert briefing.period == Period.daily
    assert len(briefing.items) == 3
    assert briefing.overview != ""
    assert briefing.meta.persona == "ainikki-v1"
    assert briefing.meta.rubric_version == "v1"
    assert briefing.meta.guardrails_version == "v1"
    assert set(briefing.meta.models_used.keys()) == {"cluster", "filter_topic", "score", "compose", "overview"}
    assert briefing.meta.models_used["compose"]


def test_pipeline_raises_on_critical_score_failure(tmp_path):
    """If score step produces a broken response, the entire pipeline crashes —
    NO file is written with an arbitrary selection."""
    tmp_out = tmp_path / "out"
    tmp_data = tmp_path / "data"
    raw_items = _load_test_raw_items()

    def broken_score_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        is_anthropic = request.url.host == "api.anthropic.com"
        system_prompt = body.get("system", "") if is_anthropic else body["messages"][0]["content"]

        if "arvioi päivän ehdokaslistaa" in system_prompt:
            content = "tämä ei ole json:ia ollenkaan"
        elif "uutisanalyytikko" in system_prompt:
            user_content = body["messages"][0]["content"] if is_anthropic else body["messages"][1]["content"]
            n = len([line for line in user_content.splitlines() if line.strip().startswith("[")])
            content = json.dumps({"clusters": [
                {"candidate_indices": [i], "primary_index": 0, "reason": None} for i in range(n)
            ]})
        else:
            content = "{}"

        if is_anthropic:
            return httpx.Response(200, json={"content": [{"type": "text", "text": content}]},
                                   headers={"x-request-id": "test"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.0002},
        })

    llm_client = httpx.Client(transport=httpx.MockTransport(broken_score_handler))
    since = datetime(2026, 7, 16, tzinfo=timezone.utc)
    until = since + timedelta(days=1)

    with pytest.raises(ScoreValidationError):
        run_pipeline(
            topic="ai", period=Period.daily, since=since, until=until,
            config_dir=Path("config"), data_dir=tmp_data, out_dir=tmp_out,
            raw_items_override=raw_items, llm_client=llm_client,
        )
    assert not tmp_out.exists() or not any(tmp_out.iterdir()), \
        "after critical failure there must be no written file"


def test_pipeline_raises_on_empty_collect(tmp_path):
    """If collect step produces no items (e.g. all sources fail),
    the pipeline proceeds through dedup/cluster/score with an empty list
    and ultimately hits EmptyBriefingError in Validate — no empty digest
    is written."""
    tmp_out = tmp_path / "out"
    tmp_data = tmp_path / "data"
    since = datetime(2026, 7, 16, tzinfo=timezone.utc)
    until = since + timedelta(days=1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({"clusters": []})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.0002},
        })

    llm_client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(EmptyBriefingError):
        run_pipeline(
            topic="ai", period=Period.daily, since=since, until=until,
            config_dir=Path("config"), data_dir=tmp_data, out_dir=tmp_out,
            raw_items_override=[], llm_client=llm_client,
        )

