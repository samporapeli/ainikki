# Multi-source collection and new topic candidates

Status: proposed, not started. Tracked in ROADMAP.md under "Next".

## Problem

`agent/collect/hn.py` is the only collect adapter. This has two effects:

1. **Blocks new topics.** A topic's daily story volume on Hacker News
   varies enormously by subject. Some topics (see below) simply don't
   have enough HN coverage to fill `items_per_briefing.min` reliably.
2. **Weakens the "source independence" scoring criterion.** The rubric
   (`config/rubrics/*.yaml`) has a "lähteiden riippumattomuus" criterion
   that assumes multiple candidates on the same story imply independent
   confirmation. In practice, multiple HN submissions of the same link
   are not independent sources — they're the same platform's users
   resubmitting the same thing. Real independence requires stories to
   originate from genuinely separate outlets.

## Proposed direction

- Add `agent/collect/rss.py` alongside `hn.py`, implementing the same
  "fetch raw, parse pure function" split used in `hn.py` for testability
  (`fetch_rss_raw()` / `parse_rss_entries()`).
- `config/sources/{topic}.yaml` (the directory already exists, empty —
  it was clearly anticipated but never filled in) lists which adapters
  and feed URLs apply to a given topic, e.g.:

  ```yaml
  topic: linux
  sources:
    - type: hn
      min_points: 15
    - type: rss
      url: https://fedoramagazine.org/feed/
    - type: rss
      url: https://news.opensuse.org/feed/
  ```

- `agent/pipeline.py`'s Collect step (`run_pipeline`, step 1) currently
  calls `fetch_hn()` directly — this needs to become a dispatch over the
  topic's configured sources, merging raw items from all adapters before
  Dedup. Dedup/Cluster/Score are source-agnostic already (they operate on
  `RawItem`/`Candidate`, not HN-specific data), so this should be a
  contained change to Collect only.
- `min_points` filtering is HN-specific; other adapters will need their
  own relevance filters (e.g. RSS has no point count — filtering there is
  more likely "only official/curated feeds" rather than a score
  threshold).

## Topic feasibility (today: HN-only)

Evaluated against "enough independent, dated news items per day to
reliably hit `items_per_briefing.min: 3`":

| Topic | HN-only fit | Notes |
|---|---|---|
| Linux (opensuse/fedora/arch/debian/general) | **Good** | HN has substantial, consistent Linux/kernel/distro coverage. Best candidate to pilot multi-topic support *before* the RSS adapter exists. Would still benefit from adding official distro blogs (Fedora Magazine, openSUSE News, Debian News) via RSS for announcements HN covers late or not at all. |
| Design (UI/UX/product/graphic) | Moderate | Real but thinner coverage (Show HN, typography/design-critique posts). Workable but noisier signal-to-noise than AI or Linux. RSS (Smashing Magazine, It's Nice That, Core77) would meaningfully improve this. |
| Mechanical keyboards (custom/split ergonomic builds, QMK/ZMK firmware) | Weak for daily | HN has niche Show-HN keyboard content but not daily volume. Likely needs Reddit (r/MechanicalKeyboards, r/ErgoMechKeyboards) or dedicated sources, **and/or** a weekly rather than daily cadence (see ROADMAP "Ideas": weekly/monthly digest variants) — a lot of this community's content is show-and-tell/builds, not "news" in the usual sense.
| Electronic music (releases, labels, festivals, scene) | Weak, effectively none on HN | Needs RSS/other sources regardless of cadence (e.g. Resident Advisor, XLR8R, Data Transmission). Also a good candidate for weekly cadence — release/festival news doesn't move at daily urgency. |

Other candidate topics with a similar source-availability profile:
- General FOSS/open source (broader than just Linux)
- Self-hosting / homelab
- Retro computing & hardware hacking (thematically adjacent to
  mechanical keyboards, decent HN presence)
- Space/science (strong, consistent HN volume)

## Recommended sequencing

1. Build the RSS adapter + `config/sources/` dispatch (unblocks
   everything else, also improves source independence for the existing
   AI topic).
2. Pilot second topic with **Linux** — works reasonably even before RSS
   lands, so it validates the multi-topic plumbing (persona/guardrails
   parameterization, see `planning/identity-and-voice.md`) independently
   from the source-diversity work.
3. Attempt one weekly-cadence topic (keyboards or electronic music) to
   validate the cadence-variant idea and a non-HN-primary source at the
   same time.
