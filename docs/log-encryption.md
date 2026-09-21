# Shared-key battle log format

## Security boundary

The ENC1 encryption algorithm and decryption tool are part of the project;
the real key is not. `log_secret.key` and timestamped variants such as
`log_secret.20260921.key` are ignored by Git. There is no source-code default
key, and the runtime refuses to start when its key file is missing, invalid,
or shorter than 32 UTF-8 bytes.

The submission builder deliberately copies the selected local key into the
competition archive as `FutureWarAgent/log_secret.key`, because the packaged
program needs it at runtime. Under the stated threat model, that archive must
remain available only to the competition committee. Anyone who obtains the
archive can also obtain the key and decrypt its logs.

## Create and rotate keys

From the repository root, create the first random key using only the Python
standard library:

```powershell
python -m future_war_agent.logtool generate-key
```

The command refuses to overwrite an existing key. Before rotating, preserve
the old key outside the repository if its logs may still need to be read, then
run:

```powershell
python -m future_war_agent.logtool generate-key --force
```

Use a fresh key for each competition package or match. Never paste a real key
into source code, documentation, chat, commit messages, or command-line
arguments.

## Build and run

Local development expects `log_secret.key` beside `main.py`:

```powershell
python main.py 18080
```

Build the submission with an explicit key file:

```powershell
python tools/build_submission.py dist --key-file .\log_secret.key
```

The builder validates the key and places it at the package root. The packaged
`main3.py` resolves the key relative to itself, so its behavior does not depend
on the launcher's current working directory.

## Decrypt on another computer

Transfer these items through the intended secure channel:

1. the encrypted log file;
2. the exact matching `log_secret.key` for that run;
3. `future_war_agent/seclog.py` and `future_war_agent/logtool.py` from the same
   revision.

Keep `seclog.py`, `logtool.py`, and the key file in one directory, then run:

```powershell
python logtool.py decrypt agent.log --key-file log_secret.key > decrypted.log
```

The key is read from the file and is not placed in shell history. Lines that
do not start with `ENC1:` are copied unchanged.

A suitable instruction for the AI on the other computer is:

> Keep seclog.py and logtool.py together. Run `python logtool.py decrypt
> agent.log --key-file log_secret.key > decrypted.log`. Use the supplied files
> unchanged, do not print or modify the key, and leave non-ENC1 lines as-is.

## ENC1 compatibility protocol

The original line-oriented symmetric algorithm is retained unchanged so old
ENC1 logs and existing cross-computer implementations remain compatible. Its
parameters are:

- marker: ASCII `ENC1:`;
- key: the UTF-8 contents of the supplied key file, excluding trailing CR/LF;
- IV prefix: ASCII bytes `future-war-iv`;
- counter: unsigned 64-bit big-endian, beginning at zero for every record;
- digest: SHA-256;
- ciphertext encoding: standard padded Base64.

For each formatted log record:

1. Encode the plaintext as UTF-8.
2. Calculate `SHA256(IV prefix || UTF-8 key || counter)` for counters beginning
   at zero.
3. Concatenate digests until the byte stream is at least as long as the
   plaintext.
4. Truncate the stream and XOR corresponding bytes.
5. Base64-encode the result and prepend `ENC1:`.

Decryption reverses these steps. `DEBUG`, `INFO`, and `WARNING` records are
encrypted; `ERROR` and `CRITICAL` records remain plaintext for startup and
runtime diagnosis.

ENC1 is retained for compatibility, not as a modern authenticated cipher. It
reuses a deterministic stream for each line and does not detect modified
ciphertext. Rotating and withholding the key materially improves protection
under the current threat model, but the format must not be used for passwords,
tokens, personal information, or other high-value secrets.
