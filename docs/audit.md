# Audit log

Use `agentgate audit` to inspect requests recorded by the gateway. Run it from
your agentgate checkout with the same settings and database as the gateway.
The gateway does not need to be running to read an existing database.

## Commands

```text
uv run agentgate audit tail -n 20
uv run agentgate audit stats --since 24h
uv run agentgate audit show <request-id-prefix>
```

- `tail` lists recent requests, newest first. Add `--agent ID` to filter by agent,
  or `--flagged` for requests flagged by the prompt-injection scanner or carrying
  a suspicious tool description or name.
- `stats` summarizes recorded requests. `--since` accepts hours or days, such as
  `24h` or `7d`; omit it for all recorded history.
- `show` displays a request's full record and any saved content. Supply its ID or
  a unique prefix. It searches the most recent 1,000 records. The record's
  `caveats` field lists short tags that qualify the record, such as
  `rejected:injection_blocked` on a refused request, or `truncated:classify:20000`
  when the sensitivity check stopped reading at 20,000 characters.

All three commands accept `--json` for machine-readable output.

## Select the database

`AGENTGATE_DATABASE_URL` selects the database. The default is the SQLite file
`data/agentgate.db` under the working directory; see
[audit storage](configuration.md#audit-storage).

If the gateway uses a different settings file, select that file for audit commands:

```bash
uv run agentgate --env-file /path/to/settings.env audit tail
```

For the supplied [Docker setup](docker.md), run the command inside the container:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml exec agentgate agentgate audit tail
```

The Docker database is at `/app/data/agentgate.db` in the `agentgate-data` volume.
It survives container recreation and `docker compose down`. Removing the volume
with `down --volumes` deletes the history. If the gateway is stopped, replace
`exec` with `run --rm --no-deps` in the command above.

## Example output

The [demo script](https://github.com/ccordi/agentgate/blob/main/scripts/demo.sh) displays records like these:

```text
TIME     ID       AGENT      PROVIDER MODEL          STATUS SENS    INJ     RED      COST
23:07:39 ad49fa30 demo       egress   http_post      403    secret  -       1    $0.00000
23:07:39 7d8199ec demo       mock     agentgate-demo 400    none    1.0000! 0    $0.00000
23:07:39 cea7f726 demo       mock     mock-model     200    none    0.0000  0    $0.00000
```

These rows show a denied HTTP policy check, a blocked injection and a successful
model request. `SENS` is the sensitivity class and `INJ` the injection score. `RED` is
the count of redaction matches; an HTTP policy check redacts nothing, so on its row
`RED` counts the kinds of sensitive data found. The `!` marks a score high enough to
block the request.

The demo uses a temporary database and removes it on success. Its records will
not appear in your regular gateway database afterward.

## Cost estimates

`COST` estimates USD cost from reported token counts and a bundled copy of
LiteLLM's model prices. Prices are not updated automatically; see the
[price source and refresh instructions](https://github.com/ccordi/agentgate/blob/main/src/agentgate/data/model_prices.PROVENANCE.md).

Missing prices or token usage produce zero estimated cost, so a zero does not
mean the request was free. The demo's mock model has no price entry, so its cost
is zero.

## Saved content

When `AGENTGATE_CONTENT_ENC_KEY` is set, the gateway saves the scanned content of
some requests, redacted and encrypted, for a limited time;
[audit storage](configuration.md#audit-storage) lists which requests and for how
long. To read it with `show`, set the same key. Request metadata can be read
without it.

## Missing records

Some requests leave no record: those refused by the
[`Host` header check](threat-model.md#unwanted-requests-to-the-gateway), those rejected
by the issued-key check or by admission control, and those whose connection to the
model server failed. The others are written in the background, not as part of the
response, and a write can fail under load, leaving a gap. The Prometheus metric
`agentgate_audit_write_failures_total`, served at `/metrics`, counts failed writes; the
gateway does not retry them.
