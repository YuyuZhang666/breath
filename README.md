# Future War Agent

Python 3.11 strategy service for the Future War game interface. The runtime
uses only the Python standard library.

## Start the match server

```powershell
python main.py 18080
```

The server listens on `0.0.0.0:<port>` and accepts JSON `POST` requests. A
request body may be at most 1 MiB. Every response has this shape:

```json
{"roleCommandMap": {}, "prompt": "", "executeCmd": ""}
```

Planner, parsing, or transport errors are converted to that safe response so
the process remains available for later rounds.

## Pre-match smoke test

With the server running on port `18080`:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:18080/ `
  -ContentType application/json `
  -InFile tests/fixtures/interface_request.json
```

The command should return `roleCommandMap`, `prompt`, and `executeCmd`.

## Verification

```powershell
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q future_war_agent tests main.py
```

## Offline replay

```powershell
python -m future_war_agent.evaluation path/to/replay.json
```

Replay evaluation is offline only and is not called from the match request
path.

## Encrypted battle logs

Routine logs at `DEBUG`, `INFO`, and `WARNING` use the line-oriented `ENC1`
shared-key format. `ERROR` and `CRITICAL` records remain plaintext so startup
and runtime failures can still be diagnosed without the decryption tool.

Decrypt a captured log with the project default key:

```powershell
python -m future_war_agent.logtool decrypt path/to/agent.log
```

The two portable files `future_war_agent/seclog.py` and
`future_war_agent/logtool.py` can also be copied to another computer and run
directly:

```powershell
python logtool.py decrypt path/to/agent.log
```

See `docs/log-encryption.md` for the exact format and key-sharing procedure.

## Conservative interface fallbacks

- Inventory item matching is case-insensitive, while emitted action names use
  the configured official name such as `Medicine`.
- Missing robot `targetTeam` or weapon `cooldown` remains unknown. Night
  simulation then falls back to the deterministic Phase 2 policy instead of
  inventing combat facts.
