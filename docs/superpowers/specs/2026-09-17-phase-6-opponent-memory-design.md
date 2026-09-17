# Phase 6 Opponent Belief and Match Memory Design

## 1. Objective

Phase 6 adds bounded process-local knowledge about visible opponent units and
completed robot-wave assessments. It preserves useful evidence across normal
strategy-session discontinuities and side swaps without inventing hidden facts.

## 2. Evidence model

Every opponent track is immutable and contains unit ID, role type, normalized
position, health, last-seen round, and integer confidence from 0 to 100.
Currently visible evidence has confidence 100. Unseen tracks decay by a fixed
amount per elapsed round and are deleted at zero. A visible unit with an
existing ID replaces contradictory type, position, and health immediately.

Unknown role strings and malformed semantic values are stored as observations
but never converted into combat claims. The belief holds at most 64 tracks,
ordered by confidence, recency, and stable ID.

## 3. Side normalization

Memory uses a canonical orientation. If our living station starts in the upper
or left half of the map, positions are rotated 180 degrees; otherwise they are
kept. Rotation uses `(width - 1 - x, height - 1 - y)` and is applied to enemy
units and static zone facts. Missing station evidence falls back to identity.

This rule makes mirrored first/second-half observations comparable while
remaining deterministic and reversible.

## 4. Match memory

`MatchMemory` stores team ID, last observed round, bounded opponent belief,
normalized static zones, and at most 16 wave summaries. Each summary records
the certificate classification, secured bit, station survival probability, and
worst station health for the round that produced it.

`MatchMemoryStore` is separate from `SessionStore`. Session discontinuity resets
short-lived planner/director/task state but does not erase match memory. A new
team ID receives independent empty memory. Nothing is written to disk.

## 5. Engine integration and safety

The locked engine updates memory once for every non-duplicate observation and
again when a fresh Phase 3 certificate is produced. Memory-update exceptions
are atomic and non-fatal: planning continues with the existing immutable
memory. Duplicate responses remain byte-equivalent and do not decay or append.

Phase 6 exposes a conservative opponent signal for later directors, but does
not enable pressure, scouting, summoning, or any experimental rule by default.
Current visible danger and Phase 3 simulation remain authoritative.

## 6. Acceptance criteria

- visible observations create confidence-100 tracks;
- missing observations decay and zero-confidence tracks disappear;
- contradictory sightings replace old facts deterministically;
- tracks, zones, and wave summaries are bounded;
- mirrored side observations normalize to the same coordinates;
- match memory survives a strategy-session discontinuity and side swap;
- duplicate requests do not mutate memory;
- empty/corrupt optional evidence falls back safely;
- the complete suite, compile check, diff check, and independent review pass.
