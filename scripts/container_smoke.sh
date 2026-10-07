#!/usr/bin/env bash
# Build and exercise the documented Docker setup: initialize on the host, stream
# through the mock, inspect audit records, and recreate containers without losing
# configuration or history. Needs Docker/Compose and curl, not host Python or models.
#
# Uses a temporary config directory, an isolated Compose project and host port
# 14100 (SMOKE_PORT to override). Removes only this run's containers and volumes.
# First build may pull base images and dependencies. Run on a host that supports
# Docker's loopback port publishing. The obs and otel profiles are checked by
# tests/test_deploy_posture.py and are not started here.
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE_PORT="${SMOKE_PORT:-14100}"
# Do not inherit the caller's live gateway configuration or Compose profiles.
for setting in ${!AGENTGATE_@}; do unset "$setting"; done
unset COMPOSE_PROFILES
scratch_dir="$(mktemp -d "${TMPDIR:-/tmp}/agentgate-smoke.XXXXXX")"
COMPOSE=(docker compose --env-file "$scratch_dir/.env" -f deploy/docker-compose.yml
         -p "agentgate-smoke-$$" --profile gateway)
export AGENTGATE_PUBLISH_PORT="$SMOKE_PORT"
cleanup() {
  if [ -f "$scratch_dir/.env" ]; then
    "${COMPOSE[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
  fi
  rm -rf "$scratch_dir"
}
trap cleanup EXIT

printf '%s\n' '== build (first build may download images and dependencies)'
docker compose --env-file /dev/null -f deploy/docker-compose.yml build agentgate
printf '%s\n' '== initialize host configuration through the image'
docker run --rm --pull never --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$scratch_dir,dst=/config" \
  agentgate:dev agentgate init --directory /config
[ -r "$scratch_dir/.env" ] || { echo 'FAIL: host cannot read generated configuration'; exit 1; }
cp "$scratch_dir/.env" "$scratch_dir/original.env"
if docker run --rm --pull never --network none --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$scratch_dir,dst=/config" \
    agentgate:dev agentgate init --directory /config; then
  echo 'FAIL: initialization overwrote an existing file'; exit 1
fi
cmp "$scratch_dir/.env" "$scratch_dir/original.env"
docker run --rm --pull never --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$scratch_dir,dst=/config,readonly" \
  agentgate:dev python -c '
import os, pathlib, stat
p = pathlib.Path("/config/.env")
assert p.stat().st_uid == os.getuid(), "unexpected owner"
assert stat.S_IMODE(p.stat().st_mode) == 0o600, "unexpected permissions"
print("PASS: configuration survives initialization, is owner-only, and is not overwritten")'

wait_ready() {
  for _ in $(seq 1 60); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${SMOKE_PORT}/readyz" >/dev/null 2>&1; then return; fi
    sleep 1
  done
  echo 'FAIL: gateway did not become ready'
  "${COMPOSE[@]}" logs agentgate
  exit 1
}

printf '%s\n' '== start and send a streaming request'
"${COMPOSE[@]}" up -d
wait_ready
curl -fsS --max-time 30 -N -D "$scratch_dir/headers" -o "$scratch_dir/body" \
  -H 'Authorization: Bearer container-smoke' -H 'Content-Type: application/json' \
  "http://127.0.0.1:${SMOKE_PORT}/v1/chat/completions" \
  -d '{"model":"mock-model","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
grep -qi '^content-type: text/event-stream' "$scratch_dir/headers"
grep -q 'data: \[DONE\]' "$scratch_dir/body"
grep -q '"content": "Hello"' "$scratch_dir/body"
printf '%s\n' 'PASS: mock response through the gateway'

cid="$("${COMPOSE[@]}" ps -q agentgate)"
bindings="$(docker port "$cid")"
[ -n "$bindings" ] || { echo 'FAIL: no published ports'; exit 1; }
if printf '%s\n' "$bindings" | grep -vqE '^[0-9]+/(tcp|udp) -> 127\.0\.0\.1:[0-9]+$'; then
  echo 'FAIL: non-loopback published port'; exit 1
fi
printf '%s\n' 'PASS: loopback-only published ports'

for _ in $(seq 1 10); do
  "${COMPOSE[@]}" exec -T agentgate agentgate audit tail --json > "$scratch_dir/before.json"
  if grep -q '"status": 200' "$scratch_dir/before.json"; then break; fi
  sleep 1
done
grep -q '"status": 200' "$scratch_dir/before.json"
printf '%s\n' '== recreate containers, keeping their audit volume'
"${COMPOSE[@]}" down
"${COMPOSE[@]}" up -d
wait_ready
"${COMPOSE[@]}" exec -T agentgate agentgate audit tail --json > "$scratch_dir/after.json"
cmp "$scratch_dir/.env" "$scratch_dir/original.env"
docker run --rm --pull never --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$scratch_dir,dst=/check,readonly" \
  agentgate:dev python -c '
import json, pathlib
p = pathlib.Path("/check")
before = json.loads((p / "before.json").read_text())
after = json.loads((p / "after.json").read_text())
assert before and any(r["status"] == 200 and r["route_provider"] == "mock" for r in before)
assert {r["id"] for r in before} <= {r["id"] for r in after}, "audit history lost"
print("PASS: audit history and credentials survive container recreation")'
printf '%s\n' 'SMOKE PASS'
