# Future War Agent

Python 3.11 strategy service for the Future War game interface. The runtime
uses only the Python standard library.

## Start the match server

Create the local log key before the first run. The generated
`log_secret.key` is ignored by Git:

```powershell
python -m future_war_agent.logtool generate-key
```

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

There is no key embedded in the source code. The server refuses to start if
`log_secret.key` is missing, empty, or shorter than 32 UTF-8 bytes. Rotate it
before a new match or submission with:

```powershell
python -m future_war_agent.logtool generate-key --force
```

Save the previous key separately before rotation if old logs may still need
to be decrypted.

Decrypt a captured log with its matching key file:

```powershell
python -m future_war_agent.logtool decrypt path/to/agent.log `
  --key-file path/to/log_secret.key
```

The two portable files `future_war_agent/seclog.py` and
`future_war_agent/logtool.py` can also be copied to another computer and run
directly:

```powershell
python logtool.py decrypt path/to/agent.log `
  --key-file path/to/log_secret.key
```

Build the competition archive with the same local key:

```powershell
python tools/build_submission.py dist --key-file .\log_secret.key
```

The archive contains the key under the fixed runtime name `log_secret.key`.
Treat the archive as secret and give it only to the competition committee.
See `docs/log-encryption.md` for the exact format, rotation, and cross-computer
handoff procedure.

## Conservative interface fallbacks

- Inventory item matching is case-insensitive, while emitted action names use
  the configured official name such as `Medicine`.
- Missing robot `targetTeam` or weapon `cooldown` remains unknown. Night
  simulation then falls back to the deterministic Phase 2 policy instead of
  inventing combat facts.
