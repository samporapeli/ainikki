# Multi-source collection and new topic candidates

Status: proposed, not started. Tracked in ROADMAP.md under "Next".

## Problem

`agent/collect/hn.py` is the only collect adapter. This has two effects:

1. **Limits which topics are viable.** Some topics get strong coverage on
   HN, others barely any — and HN's single editorial lens (tech-focused,
   English-speaking, upvote-driven) is a poor match for topics where a
   different kind of news diet is desirable (e.g. Finnish outlets for a
   domestic perspective, or enthusiast forums for a niche community).
2. **Weakens the "source independence" scoring criterion.** The rubric
   (`config/rubrics/*.yaml`) has a "lähteiden riippumattomuus" criterion
   that assumes multiple candidates on the same story imply independent
   confirmation. In practice, multiple HN submissions of the same link
   are not independent sources. Real independence requires stories to
   originate from genuinely separate outlets.

## Proposed direction

- Add new collect adapters alongside `hn.py`, each implementing the same
  "fetch raw, parse pure function" split for testability.
- `config/sources/{topic}.yaml` (the directory already exists, empty —
  it was clearly anticipated) lists which adapters and parameters apply
  to a given topic. Example:
  ```yaml
  topic: linux
  sources:
    - type: hn
      min_points: 15
    - type: rss
      url: https://fedoramagazine.org/feed/
  ```
- `agent/pipeline.py`'s Collect step dispatches over the topic's
  configured sources and merges raw items before Dedup. Dedup/Cluster/
  Score are already source-agnostic, so this is a contained change.
- Each adapter can have its own relevance filter (HN has `min_points`,
  RSS has none — filtering there is a matter of choosing curated feeds).

## Candidate sources beyond HN

These sources cover the Finnish and global news landscape at relevant
angles. Not all are suitable for all topics; the table below maps them.

### Yle (Finnish public broadcaster)

RSS available with topic-specific feeds (tiede, kulttuuri, talous,
kotimaa, etc. — see https://yle.fi/rss). Terms
(https://yle.fi/a/20-10008076): headlines may be displayed with links
back to Yle; photos and full article content may not be copied; feed
cannot be used in paid services. No paywall.

Robots.txt blocks known AI crawlers by name (GPTBot, ClaudeBot, etc.)
but permits our custom user agent via the `*` wildcard to access `/a/`
article paths. Trafilatura extraction succeeds on typical Yle articles.

### Helsingin Sanomat

Many articles are behind a paywall, which prevents summarization — out
of scope for the foreseeable future.

### Reddit (topic-specific subreddits)

Strong topical depth for niche communities (mechanical keyboards,
electronic music production, self-hosting) where HN has little
coverage. No public RSS per subreddit, but a dedicated adapter using
Reddit's JSON API is feasible. `robots.txt` may allow or disallow
scraping.

### Topic-specific RSS feeds

Many outlets offer topic-specific RSS feeds that can be plugged into
the generic RSS adapter once it exists. This is where the main
diversification happens — it's a catalog problem, not an
infrastructure problem, after the adapter is built.

## Topic fit with suggested sources

The framing is "what sources make sense" rather than "how well does HN
fit", since HN is available for every topic as a secondary source but
shouldn't be the sole pillar for most.

| Topic | Primary sources | Notes |
|---|---|---|
| AI (existing) | HN, Yle tiede/kulttuuri | HN is strong here, but Yle adds a domestic angle AI stories often lack on HN. |
| Linux & FOSS | HN, official distro/blogs RSS (Fedora Mag, openSUSE News, Debian News, LWN.net), Reddit r/linux | HN coverage is good; distro blogs fill gaps for official release announcements. |
| Design | HN (Show HN for tools), Reddit r/Design, topic-specific feeds (Smashing Mag, A List Apart, It's Nice That) | Design RSS is the main improvement; HN alone is thin. |
| Mechanical keyboards | Reddit r/MechanicalKeyboards, r/ErgoMechKeyboards, r/olkb, Geekhack | HN has some Show HN content but very little volume. Likely works best as weekly cadence. |
| Electronic music | Resident Advisor, XLR8R, Data Transmission (RSS), Reddit r/electronicmusic, r/synthesizers | Virtually no HN coverage. Straightforward as weekly cadence with RSS feeds. |

## Recommended sequencing

1. Build the RSS adapter + `config/sources/` dispatch (unblocks source
   diversity for the existing AI topic and every new topic).
2. Add Yle feeds (low effort, no legal concerns, improves the existing
   AI digest immediately).
3. Pilot a second topic — **Linux** is lowest risk since HN already
   provides decent coverage and the persona/guardrails parameterisation
   is the main thing to validate.
4. Attempt one weekly-cadence topic (keyboards or electronic music) to
   validate the cadence-variant idea and a non-HN-primary source at the
   same time.
