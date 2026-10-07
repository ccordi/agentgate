#!/usr/bin/env bash
# Run the full test suite against Postgres 16 in a container, as a single
# self-contained command:
#
#   bash scripts/test_postgres.sh [pytest args…]
#
# It: (1) starts a disposable postgres:16 container (replacing any leftover from
# an aborted run, so reruns are idempotent), (2) waits for readiness with a real
# query — pg_isready alone can catch initdb's temporary server, (3) runs the
# suite with AGENTGATE_TEST_DATABASE_URL pointed at the container
# (tests/conftest.py `audit_db_url` creates and drops one database per test),
# (4) removes the container on exit, pass or fail.
#
# Without AGENTGATE_TEST_DATABASE_URL the suite runs on SQLite.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONTAINER=agentgate-test-pg
PORT=5433                  # unusual host port — never collides with a real postgres
PASSWORD=agentgate-test    # throwaway; the container binds loopback only

# (1) Fresh container, replacing any leftover holder of the name/port.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
echo "test_postgres: starting ${CONTAINER} (postgres:16 on 127.0.0.1:${PORT})…"
# Host network with postgres itself bound to loopback:PORT — not a bridged
# `-p` publish: on a host whose firewall drops host→172.17.0.0/16 (as a strict
# egress policy may), a docker-proxy'd port accepts the TCP connect but the
# postgres handshake black-holes. Loopback is not affected, and this binds nothing
# beyond 127.0.0.1.
docker run -d --rm --name "$CONTAINER" \
  --network host \
  -e POSTGRES_PASSWORD="$PASSWORD" \
  postgres:16 -c listen_addresses=127.0.0.1 -c port="$PORT" >/dev/null
trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT

# (2) Readiness: require an actual authenticated query over the same loopback
# TCP path the tests use, not just a listening socket. (The Unix-socket form
# would miss listen_addresses/port misconfig — and with `-c port` the socket
# moves to ${PORT} too.) ~30 s budget.
ready=""
for i in $(seq 1 60); do
  if docker exec -e PGPASSWORD="$PASSWORD" "$CONTAINER" \
      psql -h 127.0.0.1 -p "$PORT" -U postgres -q -c "SELECT 1" >/dev/null 2>&1; then
    ready=1
    echo "test_postgres: ready after ${i} check(s)."
    break
  fi
  sleep 0.5
done
if [[ -z "$ready" ]]; then
  echo "test_postgres: FAILED — postgres not ready within timeout. Container log tail:" >&2
  docker logs --tail 20 "$CONTAINER" >&2 || true
  exit 1
fi

# (3) The one env var. conftest reads it at import time and gives every test its
# own freshly created database on this server.
export AGENTGATE_TEST_DATABASE_URL="postgresql+asyncpg://postgres:${PASSWORD}@127.0.0.1:${PORT}/postgres"
echo "test_postgres: running suite against ${AGENTGATE_TEST_DATABASE_URL}"
uv run --extra guard --extra pg pytest "$@"
