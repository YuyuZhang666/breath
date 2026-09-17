# Phase 5 Task Agent Design

## 1. Objective

Phase 5 adds bounded, deterministic task acquisition and answer delivery on top
of the Phase 4 strategy engine. The task path may improve score only when the
active `StrategicIntent` permits it; survival behavior always keeps priority.

## 2. Safety boundary

- Only a living pioneer may move toward, accept, or answer a task.
- `SURVIVE` and every intent with `allow_tasks=False` suppress task overlays.
- Existing non-pioneer commands are retained. A pioneer overlay is applied only
  when its movement is in bounds and does not conflict with retained movement.
- Prompt generation is request-local and deterministic. Phase 5 never calls an
  external model or executes a command inside the request path.
- `executeCmd` remains empty because the protocol contains no confirmed sandbox
  command contract.
- Treasure summoning remains disabled. News is recorded as bounded observation
  context but cannot authorize inventory use or a summon.

## 3. Task state

`TaskAgentState` is immutable and stored in the existing locked
`StrategySession`. It records the selected task type, acceptance round,
deadline, last prompt fingerprint, a pending submitted answer, and at most 16
successful SOP entries. State is reset on session discontinuity.

An SOP is learned only when the immediately preceding Phase 5 decision
submitted an answer and the next observation explicitly reports success for
that pioneer. Failed, missing, or ambiguous results never become SOPs.

## 4. Acquisition and valuation

When no task is active, Phase 5 filters task points to `is_valid`, zero
cooldown, reachable by the pioneer, and a positive reward. Candidates use a
stable value combining score reward, gold reward, path cost, and timeout
urgency. Stable ties use task type and coordinates.

The selected pioneer moves one legal step toward task interaction. Once within
Chebyshev distance one, it emits `acceptTask`. At most one task action is added
per turn.

## 5. Prompt and answer lifecycle

When `phaseTask` is nonblank:

1. A successful exact-type SOP is submitted immediately.
2. Otherwise, a nonblank `llmResp` is normalized, length-bounded, and submitted.
3. Otherwise, a deterministic prompt requests a concise answer-only response
   and preserves all safe base commands except the pioneer command.

If a bounded best answer exists at the last known deadline round it is
submitted instead of requesting more work. Duplicate observations still use
the engine cache, so prompts and submissions are byte-equivalent.

## 6. Integration and fallback

The engine computes the Phase 2/3 decision first, then invokes `TaskAgent` with
the current intent and previous task state. A task-agent exception is atomic:
the base decision is returned and an empty task state is stored. Final Phase 1
validation remains authoritative.

## 7. Acceptance criteria

- the nearest high-value valid task is selected deterministically;
- the pioneer moves, accepts, prompts, and submits through validated actions;
- successful answers become bounded SOPs and failed answers do not;
- task work is interrupted under a survival intent;
- prompt/answer size is bounded and `executeCmd` stays empty;
- no treasure action is emitted without a future confirmed contract;
- duplicate requests remain byte-equivalent and the complete suite passes.
