"""
Ajo: python -m tests.test_pipeline (projektin juuresta)

Tämä on integraatiotesti: koko run_pipeline() ajetaan päästä päähän
OIKEILLA tuotantokonfiguraatioilla (config/models.yaml, config/personas/,
config/guardrails/, config/rubrics/, config/golden_examples/) - vain
verkkokutsut (LLM + artikkelisisällön haku) on korvattu mock-transportilla.

Tämä on arvokkaampi kuin yksikkötestit siksi että se todistaa moduulien
KYTKENNÄN toimivan (oikeat parametrit, oikeat tiedostopolut, oikeat
tyyppimuunnokset vaiheiden välillä) - yksikkötestit todistavat vain että
kukin moduuli toimii eristyksissä.
"""

import json
import tempfile
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from agent.pipeline import run_pipeline
from agent.score import ScoreValidationError
from agent.validate import EmptyBriefingError
from agent.collect.hn import parse_hn_hits
from agent.schema import Period, Briefing

CLEAN_HTML = Path("tests/fixtures/article_clean.html").read_text()


def _load_test_raw_items():
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    return parse_hn_hits(fixture["hits"])  # 4 itemiä


def _make_llm_handler(score_min=3, score_max=10):
    """Yksi handler joka palvelee kaikkia neljää LLM-stepiä (cluster/score/
    compose/overview), jotka voivat molemmat käyttää eri provideria
    (OpenAI-yhteensopiva TAI Anthropic, ks. config/models.yaml: compose
    käyttää anthropic/claude-sonnet-4-6:ta muiden käyttäessä OpenRoutria).
    Step tunnistetaan system promptin sisällöstä.
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
        elif "valitsee päivän tärkeimmät" in system_prompt:
            k = max(score_min, min(n, score_max))
            selected = [{"candidate_index": i, "rank": i + 1, "selection_reason": f"peruste {i}"}
                        for i in range(k)]
            content = json.dumps({"selected": selected})
        elif "AI-uutiskoosteen toimittaja" in system_prompt:
            content = json.dumps({"headline": "Testiotsikko juttu",
                                   "summary": "Testiyhteenveto joka kuvaa juttua lyhyesti."})
        elif "julkaistuista" in system_prompt:
            content = json.dumps({"overview": "Päivän aiheet liittyivät tekoälyyn ja sen kehitykseen."})
        else:
            raise AssertionError(f"tunnistamaton system prompt: {system_prompt[:100]}")

        if is_anthropic:
            return httpx.Response(200, json={"content": [{"type": "text", "text": content}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return handler


def _enrich_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=CLEAN_HTML)


def test_full_pipeline_happy_path(tmp_out=None, tmp_data=None):
    if tmp_out is None:
        tmp_out = Path(tempfile.mkdtemp())
    if tmp_data is None:
        tmp_data = Path(tempfile.mkdtemp())
    try:
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
        assert len(briefing.items) == 4
        assert briefing.overview != ""
        assert briefing.meta.persona == "ainikki-v1"
        assert briefing.meta.rubric_version == "v1"
        assert briefing.meta.guardrails_version == "v1"
        assert set(briefing.meta.models_used.keys()) == {"cluster", "score", "compose", "overview"}
        assert briefing.meta.models_used["compose"] == "openrouter/anthropic/claude-sonnet-4-6"
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)
        shutil.rmtree(tmp_data, ignore_errors=True)


def test_pipeline_raises_on_critical_score_failure(tmp_out=None, tmp_data=None):
    """Jos score-step tuottaa rikki menevän vastauksen, koko putki kaatuu -
    EI kirjoiteta tiedostoa mielivaltaisella valinnalla."""
    if tmp_out is None:
        tmp_out = Path(tempfile.mkdtemp())
    if tmp_data is None:
        tmp_data = Path(tempfile.mkdtemp())
    try:
        raw_items = _load_test_raw_items()

        def broken_score_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            is_anthropic = request.url.host == "api.anthropic.com"
            system_prompt = body.get("system", "") if is_anthropic else body["messages"][0]["content"]

            if "valitsee päivän tärkeimmät" in system_prompt:
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
                return httpx.Response(200, json={"content": [{"type": "text", "text": content}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        llm_client = httpx.Client(transport=httpx.MockTransport(broken_score_handler))
        since = datetime(2026, 7, 16, tzinfo=timezone.utc)
        until = since + timedelta(days=1)

        try:
            run_pipeline(
                topic="ai", period=Period.daily, since=since, until=until,
                config_dir=Path("config"), data_dir=tmp_data, out_dir=tmp_out,
                raw_items_override=raw_items, llm_client=llm_client,
            )
            assert False, "olisi pitänyt heittää ScoreValidationError"
        except ScoreValidationError:
            assert not tmp_out.exists() or not any(tmp_out.iterdir()), \
                "kriittisen virheen jälkeen EI saa olla kirjoitettua tiedostoa"
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)
        shutil.rmtree(tmp_data, ignore_errors=True)


def test_pipeline_raises_on_empty_collect(tmp_out=None, tmp_data=None):
    """Jos collect-vaihe ei tuota yhtään itemiä (esim. kaikki lähteet epäonnistuvat),
    putki etenee dedup/cluster/score:aan tyhjällä listalla ja päätyy lopulta
    EmptyBriefingError:iin Validate-vaiheessa - ei kirjoiteta tyhjää koostetta."""
    if tmp_out is None:
        tmp_out = Path(tempfile.mkdtemp())
    if tmp_data is None:
        tmp_data = Path(tempfile.mkdtemp())
    try:
        since = datetime(2026, 7, 16, tzinfo=timezone.utc)
        until = since + timedelta(days=1)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"clusters": []})}}]})

        llm_client = httpx.Client(transport=httpx.MockTransport(handler))

        try:
            run_pipeline(
                topic="ai", period=Period.daily, since=since, until=until,
                config_dir=Path("config"), data_dir=tmp_data, out_dir=tmp_out,
                raw_items_override=[], llm_client=llm_client,
            )
            assert False, "olisi pitänyt heittää EmptyBriefingError"
        except EmptyBriefingError:
            pass
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)
        shutil.rmtree(tmp_data, ignore_errors=True)


if __name__ == "__main__":
    test_full_pipeline_happy_path()
    test_pipeline_raises_on_critical_score_failure()
    test_pipeline_raises_on_empty_collect()
    print("\nKaikki testit läpi.")
