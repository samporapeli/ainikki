# AGENTS.md

## Kielen ja kommenttien säännöt

- **Älä kommentoi nimiä.** Tiedoston, funktion tai muun nimikkeen nimen pitäisi puhua puolestaan. Jos nimi vaatii selityksen, se on huono nimi - valitse parempi nimi sen sijaan.
- Älä kommentoi triviaaleja asioita (esim. `set -euo pipefail` ei tarvitse kommenttia).
- Kommentit ovat "why", ei "what". Jos "what" ei ole itsestäänselvä koodista, koodi on liian vaikeaa.

## Koodityyli

- Python 3.13, ei pytest:ä - pelkät `def test_*()` + `if __name__ == "__main__"` + `python -m tests.test_<name>`
- Kaikki promptit ja config suomeksi, koodikomentit suomeksi
- Ei turhia dependencies: httpx, ruamel.yaml, click, openai - kaikki jo `requirements.txt`:ssä
- Ei emojisia tai turhia koristeita koodeissa

## Pipeline

Collect → Dedup → Cluster → Score → Enrich → Compose → Overview → Validate → Write

- Failure policy: cluster = graceful degradation, score = hard failure, compose = per-item
- OpenRouter LLM, paid models required
- `strip_code_fences()` kaikille LLM-vastauksille ennen `json.loads()`
- `response_format: {"type": "json_object"}` OpenAI-yhteensopivissa pyynnöissä

## Sivusto

Astro, staattinen HTML, zero JS. Skripti `pipeline.sh` = data-pipeline, ei sivustobuildi.
