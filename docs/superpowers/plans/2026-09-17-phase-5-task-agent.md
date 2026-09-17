# Phase 5 Task Agent Implementation Plan

**Goal:** Add safe task acquisition, deterministic prompting, bounded answer
submission, and successful-answer SOP reuse without weakening survival logic.

**Architecture:** A pure task module evaluates observations and returns a
decision overlay plus immutable state. `StrategyEngine` owns that state inside
its existing lock and treats failures atomically.

## Engineering rules

- Develop each behavior test-first and observe the expected failure.
- Preserve the one-action-per-role rule and final validator boundary.
- Use only Python 3.11-compatible standard-library features.
- Keep external calls and shell execution outside the request path.
- Run focused tests after every task and the complete suite before review.

### Task 1: Extend test fixtures and define task contracts

- Extend `tests/strategy_helpers.py` with optional tasks, task text, LLM output,
  action results, news, and score fields.
- Add immutable `TaskAgentState`, `TaskSop`, and `TaskAgentResult` contracts.
- Test state validation, bounded normalization, and deterministic task scoring.

### Task 2: Acquire reachable tasks

- Select one valid zero-cooldown task by stable reward/travel/deadline value.
- Move one safe pioneer step or emit `acceptTask` when adjacent.
- Preserve non-pioneer base commands and reject movement conflicts.
- Test invalid, unreachable, tied, adjacent, and interrupted cases.

### Task 3: Prompt, submit, and learn SOPs

- Generate a deterministic bounded answer-only prompt for active task text.
- Sanitize and submit nonblank `llmResp` through the pioneer.
- Learn an exact-type SOP only from explicit successful action feedback.
- Bound SOP count and reuse a successful answer without another prompt.
- Keep `executeCmd` empty and never emit treasure actions.

### Task 4: Integrate the locked engine

- Store task state in `StrategySession`.
- Apply task overlays after the base Phase 2/3 decision.
- Reset task state on discontinuity and atomically return the base decision on
  task-agent failure.
- Preserve duplicate request caching and deterministic serialization.

### Task 5: Acceptance, review, and merge

- Exercise move -> accept -> prompt -> submit -> successful SOP reuse through
  the real engine and final validator.
- Verify survival interruption, failure fallback, bounded state, and no command
  or treasure output.
- Run full tests, compileall, diff checks, and independent code review.
- Commit Phase 5, fast-forward merge it to local `master`, retest, and delete
  the merged branch.
