# Identity and voice: what "Ainikki" claims to be

Status: partially implemented, one open product decision remains (see
"Open question" at the bottom).

## What changed and why

The pipeline's LLM prompts and the site UI used to frame the Compose
step as "you are the AI news digest's toimittaja [journalist/editor]
'ainikki'", and the site displayed a "Toimittaja" label next to the
model name used for that step. Two problems with this:

1. It overstated the nature of the task. Composing a 2-4 sentence
   summary from one already-selected source, under a fixed rubric and
   guardrails, is not equivalent to journalism (reporting, verification,
   editorial judgment under uncertainty). A self-description should
   match what the system actually does, not what it sounds like it does.
2. It put the *model* front and center as if the model were the named
   role ("Toimittaja — claude-sonnet-4-6"), when the actual author-facing
   identity should be Ainikki, with the model as an implementation
   detail.

The fix: system prompts state the task directly without role-play,
model implementation details moved to a colophon at the page bottom,
and Ainikki is credited as author in the header.

## Open question: one Ainikki, or many personas under Ainikki?

Today there is exactly one topic (`ai`) and one persona file
(`config/personas/ainikki_v1.yaml`), loaded unconditionally regardless
of topic (`ConfigPaths.persona` in `agent/pipeline.py` is not
topic-parameterized, unlike `ConfigPaths.rubric`). The persona's
`target_audience` is explicitly "kokenut ohjelmistokehittäjä" — a voice
tuned for the current AI-news audience.

Adding a topic with a genuinely different audience (design, Linux,
mechanical keyboards, electronic music — see
`planning/multi-source-and-topics.md`) raises a real question: is
"Ainikki" one consistent author-voice across every topic, or does each
topic get its own named voice under an "Ainikki" umbrella brand?

**Option A — One universal Ainikki voice everywhere.**
Same persona (tone, target audience framing) regardless of topic; only
the rubric (what counts as newsworthy) changes per topic.
- Pros: simplest, strongest single-brand recognition, no reader
  confusion about "who's writing this".
- Cons: a voice tuned for "experienced developer reading AI news" may
  read oddly for a mechanical-keyboards or electronic-music digest.
  Forces one-size-fits-all tone across audiences that may want
  different things (e.g. more playful for keyboards vs. drier for
  Linux).

**Option B — Ainikki as umbrella brand, named sub-personas per topic.**
E.g. site/brand stays "Ainikki", but each topic's byline names its own
persona (something like "AI-uutiset, kirjoittanut Ainikki" vs. a
distinct name for the keyboards digest), similar to a publication with
multiple named columnists.
- Pros: voice can be genuinely tuned per audience; scales cleanly as
  more topics are added; matches how real multi-topic publications work.
- Cons: dilutes single-brand recognition; more naming/design surface to
  invent and maintain per topic; raises the exact same "is this persona
  also making an overstated claim" question independently for each new
  name.

**Option C — Single visible "Ainikki" author identity, persona config
parameterized per topic under the hood.**
`target_audience`/`tone_description`/`voice_traits` vary by topic
(`personas/{topic}_v1.yaml`, falling back to a shared default), but the
site always credits "Ainikki" as author — no new visible names.
- Pros: keeps the single recognizable brand from Option A while letting
  the actual writing adapt per audience; smallest UI/branding surface;
  the person is one entity, the way a human writer would properly speak
  differently to different audiences without becoming a different
  writer.
- Cons: if a reader follows two very differently-toned Ainikki digests,
  the tonal gap might feel inconsistent under one name. Needs the
  personas to still share *some* throughline (e.g. the guardrails'
  fact-based/no-speculation rules, which already apply regardless of
  persona) to stay recognizably "the same Ainikki".

**Recommendation:** start with Option C. It unblocks the concrete
technical gap (persona/guardrails not being topic-aware, see ROADMAP
"Next") without committing to a bigger branding decision, and it is the
easiest to walk back or extend into Option B later if a new topic's
audience turns out to need a genuinely distinct voice/name rather than
just a different tone-of-the-same-voice. This is a product/branding
decision — revisit once multi-source collection and a second topic
validate the need.

### If Option C is chosen, the concrete follow-up is:

- Make `ConfigPaths.persona` and `ConfigPaths.guardrails` in
  `agent/pipeline.py` topic-aware, mirroring how `ConfigPaths.rubric`
  already resolves `rubrics/{topic}_scoring_rubric_v1.yaml`:
  `personas/{topic}_v1.yaml`, with a fallback to a shared default
  persona file if a topic doesn't have its own yet.
- Guardrails likely stay shared across all topics regardless of which
  option is chosen — they're fact/reliability/language rules, not voice,
  and are already documented as persona-independent by design.