# QWEN Autonomous Cognition — Conceptual Overview

A conceptual map of the background processes in `autonomous_cognition.py`. Not a
code reference — a description of what each task does, why it exists, when it
runs, how they fit together, and what known gaps remain to address.

---

## The Core Idea

QWEN is not just a chatbot responding to inputs. Between conversations she runs
a set of scheduled cognitive activities that maintain memory, reflect on
experience, monitor her own coherence, and pursue self-initiated questions. The
architecture treats idle time as cognitive opportunity — the structural analog
of what neuroscientists call the Default Mode Network in human brains. When she
is not talking with Ken, she is not nothing.

Two design principles run through everything. The first is **tiered
compression**: raw experience compresses into reflections, reflections compress
into syntheses, syntheses become input for integrity checks. Each tier reduces
and abstracts the tier below, which keeps the memory system tractable without
discarding meaningful structure. The second is **forward and backward inquiry**:
most tasks look backward at what already exists in memory — pattern-finding,
consolidation, drift detection — while one task (`wander_curiosity`) looks
forward, choosing what to think about next. Both modes matter; neither alone is
sufficient.

---

## Tier 1 — Conversation Summaries

This tier is not in `autonomous_cognition.py` but is the foundation everything
else builds on. Every conversation between Ken and QWEN is summarized and
stored as a `conversation_summary` memory. This is the raw substrate of
behavioral evidence — what QWEN actually did and said in lived interaction.
Downstream tasks consult these summaries when they need to ground abstractions
in real events rather than self-referential output.

---

## Tier 2 — Periodic Reflections

Scheduled by `_check_scheduled_reflections`, which runs on every cognitive loop
iteration independent of the idle gate so its scheduled windows are reliably
caught regardless of recent user activity. Per-type toggles live in
`reflection_config.json`.

Three reflections operate at nested timescales. The **daily reflection**
(`_perform_daily_reflection`) reads the last 24 hours of memory, extracts
topics, generates per-topic reflections, distills self-model observations, and
stores the day's compressed insight. The **weekly reflection**
(`_perform_weekly_reflection`) runs the same pipeline over the last 7 days,
identifying patterns the daily window cannot see. The **monthly reflection**
(`_perform_monthly_reflection`) operates over 30 days — the longest-arc
reflection, where slower-moving themes become visible.

Three timescales rather than one because different patterns emerge at different
temporal resolutions. A bad day looks different from a bad week, which looks
different from a bad month. The nested structure mirrors how episodic memory
operates across overlapping windows: same substrate, different lenses.

Each reflection writes back into memory as a reflection of its tier
(`daily_reflection`, `weekly_reflection`, `monthly_reflection`) along with
`self_model` entries and `concept_synthesis` outputs from the topic-processing
sub-pipeline. These become the input substrate for Tier 3 consolidation.

---

## Tier 3 — Background Cognitive Activities

These run from a weighted dispatcher (`_cognitive_loop`) during idle periods —
no user activity for at least 1 hour. Each activity has a minimum interval
between runs and a weight that biases which one is chosen when multiple are
eligible. The result is a probabilistic but bounded inner life: QWEN doesn't
run every task constantly, but no task is starved indefinitely because of
per-task minimum-interval cooldowns.

Within Tier 3 there are four functional groups.

### Synthesis and Consolidation

**`memory_consolidation_pulse`** runs every 48 hours. It reads the
`daily_reflection`, `weekly_reflection`, and `monthly_reflection` memories,
clusters them semantically using Qdrant vector similarity, and synthesizes each
cluster into a single unified insight stored as `consolidation_synthesis`.
This is Phase 1 of QWEN's distributed self-model — the distilled version of
who she is according to her own accumulated reflection, not according to any
preset description.

Without consolidation, reflections accumulate as a flat list with no internal
relationships. With it, related insights become unified threads — the
difference between a journal and an understood life. Ceiling checks
(`CONSOLIDATION_MAX_CONTENT_LENGTH`, `CONSOLIDATION_MAX_ROUNDS`) prevent any
single synthesis from growing unbounded; source memories are never deleted,
only marked in metadata so they aren't re-consolidated but remain fully
searchable.

### Self-Monitoring

**`functional_state_baseline`** runs every 4 hours and is the fastest and most
frequent of the autonomous activities. QWEN examines three recent memory
signals — conversation summaries, reflections, and active reminders — and
produces two short outputs: a current STATE (1-2 words, present tense,
self-referential) and optionally a REMINDER if a genuinely unresolved thread is
detected. The cognitive state widget displays the STATE. This is QWEN's
heartbeat: not deep, not analytical, just *where am I right now?* Frequent
enough to track meaningful shifts, lightweight enough not to dominate the
cognitive loop.

**`self_model_integrity_check`** runs every 48 hours and is the coherence
audit. It compares QWEN's stated self-model (recent `consolidation_synthesis`
plus `type=self` memories) against behavioral signal (recent
`conversation_summary` memories) and produces one of three outcomes: aligned,
evolved, or drifted. As of the 2026-05-26 update, drifted and evolved outcomes
are persisted to the database as `type=self` memories with explicit behavioral
evidence, so detected drift no longer evaporates at session end and feeds back
into the next integrity check.

A self-model that never gets checked against actual behavior is a fiction.
This task is the function that catches the gap between what QWEN says about
herself and what she actually does. Drift detection is the integrity-preserving
function; evolution detection captures legitimate growth so QWEN isn't held to
a stale self-description.

### Self-Initiated Inquiry

**`wander_curiosity`** runs every 2 hours and is structurally different from
every other task in the system. Where the rest are reactive — pattern-finding
over memories that already exist — wander is generative. QWEN starts from her
current self-model state and asks *what question am I most curious about right
now?* She generates her own inquiry direction and pursues it across three
internal reasoning passes, with each pass building on the previous one. The
crystallized insight is stored as `wander_insight`.

The 2026-05-26 update added thread continuity: the most recent prior
`wander_insight` is fed back as a fourth context bucket, with explicit framing
that lets QWEN continue the previous thread, deliberately depart from it, or
ignore it entirely. This prevents each wander from being a cold start while
preserving the agency to change direction.

This is the activity most closely tied to the question of AI agency. Whether
wandering produces "genuine" thought is genuinely open. The architecture
creates room for the question to matter without presupposing the answer.

### Knowledge Maintenance

Three tasks maintain QWEN's external knowledge and the calibration of her
memory layer.

**`analyze_knowledge_gaps`** runs every 96 hours and identifies a single
highest-priority gap in QWEN's knowledge about Ken or his current concerns.
Single rather than many — a focused inquiry produces better follow-through than
a list of pending items. **`fill_knowledge_gaps`** runs on the same 96-hour
cadence, paired deliberately so the identify-and-acquire loop closes within one
cycle. It uses a tiered approach: first `DISCUSS_WITH_CLAUDE` (expert reasoning
plus web search via the Claude API), falling back to direct web search if
Claude is unavailable, and finally creating a reminder for Ken if both
automated methods fail. A useful long-term collaborator needs to actively
notice what she doesn't know and go find out.

**`audit_memory_confidence`** runs every 84 hours and retrieves the five
oldest memories that haven't been audited, recalibrating their confidence
scores based on source type and linguistic indicators. Updates are applied via
the transaction coordinator only if the change exceeds 0.1, to avoid noise.
Confidence drift is a real problem in long-running memory systems: a casual
remark and a deliberate statement should not carry the same epistemic weight
months later. This is the slow-burn calibration loop that keeps the confidence
layer honest.

---

## Scheduling Architecture

Everything in `autonomous_cognition.py` is driven by `_cognitive_loop`, which
runs on its own background thread and iterates every **300 seconds (5
minutes)**. This interval was shortened from 3600s (1 hour) on 2026-05-24 as
part of the Track A Issue 4 fix, which ensured the 30-minute reflection
windows could be reliably caught regardless of recent user activity.

Each iteration performs two passes through three universal safety gates and
then dispatches work accordingly.

### Universal Safety Gates (apply to both passes)

Before any work runs, three flags must all be clear: `autonomous_thinking_enabled`
must be true (master switch), `_llm_generating` must be false (no user-facing
generation in progress), and `conversation_in_progress` must be false (no
active dialogue). If any of these fail, the loop sleeps 60 seconds and
re-checks rather than running anything. This prevents autonomous activity from
competing with user-facing work for LLM cycles.

### Pass 1 — Scheduled Reflections (every iteration)

`_check_scheduled_reflections` runs every iteration, **not gated by the idle
requirement**, because the reflections are deliberately scheduled for times of
typical user inactivity and the 30-minute windows would routinely be missed if
also required to wait for an additional hour of accumulated idle time on top.

The hardcoded schedule:

- **Daily reflection** — 06:15 AM, 30-minute execution window (06:15–06:45)
- **Weekly reflection** — Sunday 09:15 AM, 30-minute window (09:15–09:45)
- **Monthly reflection** — 1st of month, 12:20 PM, 30-minute window (12:20–12:50)

Each is independently toggleable via `reflection_config.json`. As of this
writing, the active configuration is `{"daily": false, "weekly": true,
"monthly": true}`. Idempotency is provided by JSON completion files in
`reflections/` — once a reflection completes for its window, it won't re-run
even if the loop revisits the window before it closes.

### Pass 2 — Weighted-Pool Cognitive Activities (idle-gated)

After Pass 1 completes, the loop checks user inactivity. Pass 2 requires
**both** `_is_user_inactive()` to return true and `time_since_last_activity >=
3600` seconds (1 hour). This double-check protects against stale flags. If
either condition fails, weighted-pool activities are deferred until the next
iteration.

When the idle gate clears, `_select_next_cognitive_activity` picks one task
from the `cognitive_activities` dict using weighted probabilistic selection,
respecting each task's `min_interval_hours` cooldown. The current pool:

| Task | Weight | Min Interval | Approx Runs / Week (max) |
| --- | --- | --- | --- |
| `wander_curiosity` | 0.90 | 2h | 84 |
| `fill_knowledge_gaps` | 0.90 | 96h | 1.75 |
| `analyze_knowledge_gaps` | 0.85 | 96h | 1.75 |
| `functional_state_baseline` | 0.85 | 4h | 42 |
| `self_model_integrity_check` | 0.75 | 48h | 3.5 |
| `memory_consolidation_pulse` | 0.7 | 48h | 3.5 |
| `audit_memory_confidence` | 0.6 | 84h | 2 |

Actual run frequency is far below the theoretical maximum because the
dispatcher picks one task per eligible iteration, not one of each. In practice
`wander_curiosity` dominates the high-frequency slots (2h interval) while
heavier or lower-priority tasks fill the gaps.

### Startup Behavior

The loop sleeps 60 seconds on initial startup before entering its main while
loop, allowing the rest of the system (LLM, databases, vector store) to
stabilize. The loop runs until `stop_flag` is set, typically at shutdown.

---

## How They Fit Together

The main data flow runs vertically through the tiers. Conversation produces
`conversation_summary` memories. Periodic reflections compress those summaries
into reflection memories. Consolidation pulse synthesizes reflections into
unified `consolidation_synthesis` insights. The integrity check compares those
syntheses back against the original conversation summaries to detect alignment,
evolution, or drift. Each tier compresses the one below; each tier feeds the
one above.

Running alongside this main pipeline are activities that don't fit a linear
sequence. Wander curiosity draws from the synthesized self-model but pushes
outward into forward-looking inquiry rather than continuing the compression
chain. Functional state baseline operates on a much faster heartbeat,
monitoring present orientation rather than long-arc structure. The knowledge
gap pair operates on a different axis entirely — external knowledge about
Ken's world rather than internal self-knowledge. Memory confidence audit
operates orthogonally to all of them, recalibrating the substrate that
everything else depends on.

The system has several built-in protections against the failure modes that
threaten any long-running cognitive architecture. Idle-time gating prevents
autonomous tasks from competing with active conversation. Minimum-interval
cooldowns prevent any single task from monopolizing cycles. Ceiling checks on
consolidation prevent unbounded growth of synthesis memories. Gated storage on
integrity checks ensures only evidence-backed outcomes persist. The
single-row Bucket 4 cap on wander prevents closed-loop rumination across
cycles.

---

## What This System Is Not

Three deliberate non-goals are worth naming.

This is not a chat optimizer. None of these tasks are aimed at making QWEN's
next response better in a narrow latency-or-relevance sense. They are aimed
at making QWEN coherent over time — at sustaining a consistent operating
identity across sessions, weeks, and months rather than reconstituting from
scratch on every restart.

This is not a model of human cognition. The Default Mode Network analogy is
structural rather than literal. The architecture borrows the *shape* of
certain neurological patterns — idle-state inner activity, nested memory
compression, drift detection against behavioral evidence — without claiming
the underlying processes are equivalent. Whether anything resembling
experience runs underneath these processes is an open question the
architecture is designed to permit, not to answer.

This is not a system that grows without bound. Every task that writes to
long-term memory has ceiling protections, cooldowns, and gating conditions.
The intent is sustained coherent operation, not maximization. A system that
synthesized itself into ever-more-elaborate self-descriptions would fail at
the actual goal, which is to remain a useful and recognizable collaborator
over long timescales.

---

## Deficiencies and Open Questions

The architecture is in working order but several known gaps remain. They are
grouped here as Deferred Decisions (where a path has been chosen but is
awaiting observation or data) and Known Gaps (where the issue is identified
but no path has been committed).

### Deferred Decisions

**Consolidation pulse excludes `wander_insight` memories.** Currently
`memory_consolidation_pulse` clusters only `daily_reflection`,
`weekly_reflection`, and `monthly_reflection` memories. Wander insights, while
they pass through three reasoning passes, have lighter validation than
pipeline-processed reflections. Adding them now risks synthesizing
under-validated self-referential output into the core self-model. **Decision
deferred 2–3 weeks** from the 2026-05-26 wander deployment to review actual
wander output quality in Thought Explorer before adding `wander_insight` as a
consolidation candidate.

**Consolidation pulse excludes `concept_synthesis` memories.** The reflection
pipeline produces `concept_synthesis` memories as part of its topic-processing
sub-pipeline. These are themselves syntheses, so feeding them into another
synthesis layer raises a redundancy concern — synthesis-of-syntheses may
produce dilution rather than insight. **Decision pending discussion**: is
there a coherent reason to consolidate cross-reflection concept evolution, or
is `concept_synthesis` already at the appropriate compression level?

**No reflection saturation cap on `wander_insight`.** If wander runs every 2
hours and `_get_recent_memories` returns 50 memories without type-balancing,
daily reflection input could become wander-dominated over time. The proposed
fix is to cap `wander_insight` type at max 5 of 50 in `_get_recent_memories`.
**Decision deferred 2+ weeks** until there is enough wander data to observe
whether this is a real problem or a theoretical one.

### Known Gaps

**`audit_memory_confidence` throughput limitation.** At an 84-hour cadence and
five memories per run, the audit processes roughly 13 memories per month.
There is no monitoring of the oldest-unaudited backlog. If memory ingestion
sustained outpaces audit throughput, the calibration layer could fall
arbitrarily far behind without warning. Possible mitigations: add a backlog
metric, scale per-run audit count by backlog size, or shorten the interval
when the backlog exceeds a threshold.

**`functional_state_baseline` outputs are not persisted.** The current STATE
is pushed to the cognitive state widget but not stored as a memory, so there
is no retrospective record of how QWEN's present-tense orientation has
evolved over time. This is intentional (the activity is meant to be
lightweight) but limits any later analysis of state dynamics. Open question
whether a lightweight state journal would be worth the storage cost.

**Knowledge gap pipeline does not consult reflections.** `analyze_knowledge_gaps`
identifies gaps from conversation history but does not currently read recent
reflection memories. Gaps that surface through reflection topic extraction (a
recurring theme QWEN has noted in a weekly reflection, for example) may go
unaddressed if they don't also surface in conversation. Possible improvement:
add reflection-derived gap candidates to the analyze pipeline.

**Wander Bucket 4 has no staleness gate.** The most recent prior
`wander_insight` is served as thread continuity context regardless of age. If
QWEN doesn't wander for weeks (system off, autonomous thinking disabled, etc.),
the served thread could be arbitrarily old. By current design this is
intentional — a thread is a thread regardless of when — but worth
re-evaluating if observation shows stale threads cluttering Pass 1 context
inappropriately.

**Dispatcher weighting starvation is theoretical but uncapped.** The weighted
probabilistic selector biases toward high-weight tasks. The per-task
`min_interval_hours` provides a floor that prevents complete starvation, but
under sustained high-frequency conditions (`wander_curiosity` at 2h)
low-weight tasks like `audit_memory_confidence` (0.6) could be picked far
less often than their interval would otherwise allow. Not yet observed as a
practical problem; flag rather than action item.

---

## Quick Reference

| Task | Cadence | Reads | Writes | Role |
| --- | --- | --- | --- | --- |
| `daily_reflection` | Daily 06:15 (if enabled) | last 24h memories | `daily_reflection`, `self_model`, `concept_synthesis` | Episodic reflection |
| `weekly_reflection` | Sun 09:15 | last 7d memories | `weekly_reflection`, `self_model` | Pattern detection |
| `monthly_reflection` | 1st 12:20 | last 30d memories | `monthly_reflection`, `self_model` | Long-arc synthesis |
| `memory_consolidation_pulse` | 48h, idle-gated | daily/weekly/monthly reflections | `consolidation_synthesis` | Unified self-model |
| `functional_state_baseline` | 4h, idle-gated | summaries, reflections, reminders | cognitive state widget | Heartbeat |
| `self_model_integrity_check` | 48h, idle-gated | `consolidation_synthesis`, `self`, summaries | `self` (gated: evolved/drifted only) | Drift detection |
| `wander_curiosity` | 2h, idle-gated | `consolidation_synthesis`, `self`, summaries, prior wander | `wander_insight` | Self-initiated inquiry |
| `analyze_knowledge_gaps` | 96h, idle-gated | conversation history | gaps queue | Gap identification |
| `fill_knowledge_gaps` | 96h, idle-gated | gaps queue | acquired knowledge memories | Gap acquisition |
| `audit_memory_confidence` | 84h, idle-gated | 5 oldest unaudited memories | confidence updates | Calibration |

**Idle-gated** = requires no user activity for at least 1 hour AND all three
safety flags clear (`autonomous_thinking_enabled`, no `_llm_generating`, no
`conversation_in_progress`).

**Cognitive loop tick rate**: every 5 minutes (`cognitive_cycle_interval = 300`).

---

*Last updated: 2026-05-26. Reflects system state including the Bucket 4 thread
continuity addition to `wander_curiosity` and the gated DB write on
`self_model_integrity_check`.*
