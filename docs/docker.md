# Running with Docker

Run the gateway with demo settings and a mock model server.

You need Docker with Compose and `curl`. A host Python installation is not needed.
Run these commands from the directory where you cloned agentgate.

## Start the gateway

Build the image:

```bash
docker compose -f deploy/docker-compose.yml build agentgate
```

Create `.env` using the image. Skip this command if you already ran
`agentgate init` in this checkout:

```bash
docker run --rm --pull never --network none \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/config" \
  agentgate:dev agentgate init --directory /config
```

Start the gateway and mock model server:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml --profile gateway up -d
```

To see startup logs:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml logs agentgate
```

The ports are published only on the host's loopback interface.

## Send a request

Follow the [readiness check and chat request example](getting-started.md#send-a-request).
Then inspect the recorded request in the container:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml exec agentgate agentgate audit tail
```

The [audit guide](audit.md) explains the fields and other queries.

## Stop and clean up

Stop the containers with:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml --profile gateway down
```

Settings in `.env` are kept. Audit history is stored in the `agentgate-data` volume
and survives `down`; add `--volumes` to delete it.

## Connect a real model server

Add the provider settings as described in
[Connect a real model server](getting-started.md#connect-a-real-model-server) to `.env`.
Inside the container, `127.0.0.1` is the container itself, so give `base_url` an address
the container can reach, such as `http://host.docker.internal:8000` on Docker Desktop.
Then apply the settings with:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml up -d agentgate
```

Repeat the [chat request](getting-started.md#send-a-request) to check that your
model server responds. Then follow [Using it with an agent application](getting-started.md#using-it-with-an-agent-application)
to connect your client.

This setup uses the built-in heuristic scanner, meant only for trying the gateway
out. [Scanner models in Docker](configuration.md#scanner-models-in-docker)
covers the classifier.

← Back to [Getting started](getting-started.md).
