#!/usr/bin/env bash
# Run the full test suite with a real Redis behind the limits backend, as a single
# self-contained command:
#
#   bash scripts/test_redis.sh [pytest args…]
#
# It: (1) starts a disposable redis:7-alpine container (replacing any leftover from an
# aborted run, so reruns are idempotent), (2) waits for PING/PONG, (3) runs the suite
# with AGENTGATE_TEST_REDIS_URL pointed at the container — which un-skips the
# RedisBackend correctness tests in tests/test_redis_backend.py — and (4) removes the
# container on exit, pass or fail.
#
# Without AGENTGATE_TEST_REDIS_URL those tests skip and the suite is unchanged.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONTAINER=agentgate-test-redis
PORT=6380                  # unusual host port — never collides with a real redis

# (1) Fresh container, replacing any leftover holder of the name/port.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
echo "test_redis: starting ${CONTAINER} (redis:7-alpine on 127.0.0.1:${PORT})…"
# Host network with redis itself bound to loopback:PORT — a bridged `-p`
# publish black-holes on a host whose firewall drops host→docker-bridge traffic
# (details in scripts/test_postgres.sh).
docker run -d --rm --name "$CONTAINER" \
  --network host \
  redis:7-alpine redis-server --bind 127.0.0.1 --port "$PORT" >/dev/null
trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

# (2) Readiness: a real PING over the same loopback TCP path the tests use,
# not just a listening socket. ~15 s budget.
ready=""
for i in $(seq 1 30); do
  if [[ "$(docker exec "$CONTAINER" redis-cli -h 127.0.0.1 -p "$PORT" ping 2>/dev/null)" == "PONG" ]]; then
    ready=1
    echo "test_redis: ready after ${i} check(s)."
    break
  fi
  sleep 0.5
done
if [[ -z "$ready" ]]; then
  echo "test_redis: FAILED — redis not ready within timeout. Container log tail:" >&2
  docker logs --tail 20 "$CONTAINER" >&2 || true
  exit 1
fi

# (3) The one env var; tests/test_redis_backend.py reads it at import time.
export AGENTGATE_TEST_REDIS_URL="redis://127.0.0.1:${PORT}/0"
echo "test_redis: running suite against ${AGENTGATE_TEST_REDIS_URL}"
uv run --extra guard pytest "$@"
