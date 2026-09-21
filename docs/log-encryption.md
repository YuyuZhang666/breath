# Shared-key battle log format

## Files and commands

The runtime implementation is in `future_war_agent/seclog.py`. The matching
decryption CLI is `future_war_agent/logtool.py`.

From the repository or extracted submission package:

```powershell
python -m future_war_agent.logtool decrypt agent.log
```

For a standalone handoff, copy `seclog.py` and `logtool.py` into the same
directory on the other computer, then run:

```powershell
python logtool.py decrypt agent.log
```

An alternate shared key can be supplied with `--key`:

```powershell
python logtool.py decrypt agent.log --key "the-shared-key"
```

## ENC1 protocol

The default protocol values are:

- marker: the ASCII text `ENC1:`
- shared key: the UTF-8 text `future-war-agent-log-key-v1`
- IV prefix: the ASCII bytes `future-war-iv`
- counter: an unsigned 64-bit big-endian integer beginning at zero
- digest: SHA-256
- ciphertext encoding: standard Base64 with padding

For each complete formatted log record:

1. Encode the plaintext as UTF-8.
2. Starting with counter zero, calculate
   `SHA256(IV prefix || UTF-8 key || counter)`.
3. Increment the counter and concatenate 32-byte digests until the stream is
   at least as long as the plaintext.
4. Truncate the stream to the plaintext length and XOR corresponding bytes.
5. Base64-encode the result and prepend `ENC1:`.

Decryption removes `ENC1:`, Base64-decodes the remaining text, regenerates
the same byte stream, XORs it with the ciphertext, and decodes the result as
UTF-8. Lines without the marker are returned unchanged because error records
are deliberately left readable.

`DEBUG`, `INFO`, and `WARNING` records are encrypted. The default runtime log
level is `INFO`, so `DEBUG` records are emitted only when the configured level
is lowered. `ERROR` and `CRITICAL` records, including tracebacks, remain
plaintext. Each encrypted record is a single physical line even when its
plaintext contains embedded newlines.

## Handoff to another computer or AI

Send these items together:

1. the encrypted log file;
2. `seclog.py` and `logtool.py` from the same revision;
3. this protocol document;
4. the shared key, if it differs from the default.

The receiving AI should run the supplied tool locally instead of inventing a
different cipher. A suitable instruction is:

> Keep seclog.py and logtool.py in the same directory. Run `python logtool.py
> decrypt <log-file>` and save stdout as UTF-8. Do not modify the key, IV,
> marker, counter byte order, or Base64 variant. Leave non-ENC1 lines as-is.

## Security boundary

This format is retained for compatibility and has important limitations. It
reuses a deterministic XOR key stream for every log record and does not
authenticate ciphertext. Anyone with the shared key, or access to the source
submission containing that key, can decrypt every log. Modified ciphertext
and incorrect keys also cannot be reliably detected. Keep the key and source
package private, and do not use this format for passwords, tokens, personal
data, or other high-value secrets.
