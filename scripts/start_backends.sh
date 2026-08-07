#!/usr/bin/env bash
#
# Start Redis + Qdrant for the benchmark, with or without Docker.
#
# WHY THIS EXISTS: setup_server.sh originally only knew how to start the backends through
# `docker compose`. On a shared HPC-style node that fails in three common ways -- docker
# is not installed, the daemon is not running, or the user is not in the `docker` group --
# and the failure surfaces much later as
#
#     [ FAIL ] qdrant reachable - [Errno 111] Connection refused
#
# which matters enormously: Memora catches a missing Qdrant and silently degrades to
# Phase 1 retrieval, so the benchmark would still RUN and still produce numbers, just
# meaningless ones. Rather than leave that as a manual workaround in someone's shell
# history, this script tries Docker first and otherwise runs both services as ordinary
# user processes. No root, no container runtime, same ports, same data directories.
#
# Everything stays inside the repository:
#     .cache/bin/qdrant          downloaded static binary
#     .cache/redis-stable/       redis source + built binary (only if no redis-server)
#     .cache/run/*.pid           pidfiles for `stop`
#     data/redis/, data/qdrant/  state
#     logs/redis.log, logs/qdrant.log
#
# Usage:
#     bash scripts/start_backends.sh          # start (docker if available, else native)
#     bash scripts/start_backends.sh stop
#     bash scripts/start_backends.sh status
#     BACKEND_MODE=native bash scripts/start_backends.sh    # force native

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ACTION="${1:-start}"
RUN_DIR="$REPO_ROOT/.cache/run"
BIN_DIR="$REPO_ROOT/.cache/bin"
COMPOSE_FILE="docker-compose.benchmark.yml"

mkdir -p "$RUN_DIR" "$BIN_DIR" data/redis data/qdrant logs

# ------------------------------------------------------------------ config from .env
# Read the same variables docker-compose.benchmark.yml and src/config.py read, so all
# three agree on ports. Parsed with sed rather than `source` because .env may legally
# contain characters that would execute if sourced.
env_get() {
  local key="$1" default="$2" val=""
  if [[ -f .env ]]; then
    val="$(sed -n "s/^[[:space:]]*${key}=//p" .env | tail -1 \
           | tr -d '\r' | sed -e 's/^["'\'']//' -e 's/["'\'']$//' -e 's/[[:space:]]*$//')"
  fi
  printf '%s' "${val:-$default}"
}

REDIS_PORT="$(env_get REDIS_PORT 6379)"
QDRANT_PORT="$(env_get QDRANT_PORT 6333)"
QDRANT_GRPC_PORT="$(env_get QDRANT_GRPC_PORT 6334)"

# ------------------------------------------------------------------ readiness probes
# Probe the PROTOCOL, not just the socket. A listening port proves nothing: Qdrant accepts
# connections seconds before it will answer, and a half-started Redis would let preflight
# pass and then fail mid-ingest.

redis_ready() {
  # bash's /dev/tcp avoids depending on redis-cli, which the native path may not install.
  #
  # The braces matter: `exec 3<>file 2>/dev/null` would apply the redirect to `exec`
  # itself, permanently silencing the shell's stderr. Grouping scopes it to the group.
  local out=""
  { exec 3<>"/dev/tcp/127.0.0.1/${REDIS_PORT}"; } 2>/dev/null || return 1
  printf 'PING\r\n' >&3
  if ! read -r -t 2 out <&3; then
    { exec 3<&-; } 2>/dev/null
    return 1
  fi
  { exec 3<&-; } 2>/dev/null
  [[ "$out" == +PONG* ]]
}

qdrant_ready() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 2 "http://127.0.0.1:${QDRANT_PORT}/readyz" >/dev/null 2>&1
  else
    # Subshell so a failed connect cannot disturb this shell's descriptors.
    ( { exec 3<>"/dev/tcp/127.0.0.1/${QDRANT_PORT}"; } 2>/dev/null ) || return 1
  fi
}

wait_ready() {
  local name="$1" probe="$2" tries="${3:-60}"
  for _ in $(seq 1 "$tries"); do
    if "$probe"; then
      echo "  $name ready"
      return 0
    fi
    sleep 1
  done
  echo "  ERROR: $name did not become ready in ${tries}s - see logs/${name}.log" >&2
  return 1
}

# ------------------------------------------------------------------ docker path
docker_usable() {
  [[ "${BACKEND_MODE:-auto}" != "native" ]] || return 1
  command -v docker >/dev/null 2>&1 || return 1
  docker compose version >/dev/null 2>&1 || return 1
  # `docker info` is the check that catches a stopped daemon AND the permission-denied
  # case, both of which `command -v docker` happily passes.
  docker info >/dev/null 2>&1 || return 1
  [[ -f "$COMPOSE_FILE" ]] || return 1
}

# ------------------------------------------------------------------ native: qdrant

# Does the binary actually RUN on this host? Being executable is not enough -- the glibc
# build of Qdrant requires GLIBC_2.38 (Ubuntu 24.04+) and dies on the dynamic loader with
#     libc.so.6: version `GLIBC_2.38' not found
# on anything older. That is a loader failure, not a Qdrant failure, so it must be detected
# here rather than surfacing later as "connection refused".
qdrant_bin_works() {
  local bin="$1" out=""
  [[ -x "$bin" ]] || return 1
  out="$("$bin" --version 2>&1 || true)"
  case "$out" in
    *GLIBC*|*"not found"*|*"cannot execute"*|*"No such file or directory"*) return 1 ;;
  esac
  return 0
}

_qdrant_download() {
  local url="$1" tmp="$BIN_DIR/qdrant.tar.gz"
  echo "  fetching $url"
  curl -fL --max-time 300 -o "$tmp" "$url" || return 1
  tar xzf "$tmp" -C "$BIN_DIR" || { rm -f "$tmp"; return 1; }
  rm -f "$tmp"
  # Archive layout varies (bare binary vs nested dir); find it either way.
  if [[ ! -f "$BIN_DIR/qdrant" ]]; then
    local found
    found="$(find "$BIN_DIR" -maxdepth 3 -type f -name qdrant | head -1 || true)"
    [[ -n "$found" ]] && mv "$found" "$BIN_DIR/qdrant"
  fi
  [[ -f "$BIN_DIR/qdrant" ]] || return 1
  chmod +x "$BIN_DIR/qdrant"
}

ensure_qdrant_bin() {
  # Validate rather than just checking existence: a previously downloaded but unrunnable
  # binary (wrong libc) would otherwise be reused forever.
  if qdrant_bin_works "$BIN_DIR/qdrant"; then
    return 0
  fi
  if [[ -e "$BIN_DIR/qdrant" ]]; then
    echo "  existing qdrant binary does not run here - replacing it"
    rm -f "$BIN_DIR/qdrant"
  fi

  command -v curl >/dev/null 2>&1 || { echo "ERROR: curl required" >&2; return 1; }
  echo "  downloading qdrant"

  # Resolve assets from the releases API rather than hardcoding a filename: Qdrant has
  # changed its asset naming across releases, and a stale URL 404s in a way that looks
  # like a network problem.
  local assets=""
  assets="$(curl -fsS --max-time 20 \
             https://api.github.com/repos/qdrant/qdrant/releases/latest 2>/dev/null \
           | sed -n 's/.*"browser_download_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
           || true)"

  # MUSL FIRST, ALWAYS. The musl build is statically linked and runs on any glibc vintage;
  # the gnu build needs GLIBC_2.38+. Ordering used to be whatever the API returned first,
  # which picked gnu and broke on Ubuntu 22.04.
  local candidates=() u=""
  for pat in 'musl' 'gnu'; do
    u="$(printf '%s\n' "$assets" | grep -E "x86_64-unknown-linux-${pat}\.tar\.gz$" | head -1 || true)"
    [[ -n "$u" ]] && candidates+=("$u")
  done
  candidates+=("https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz")

  for u in "${candidates[@]}"; do
    if _qdrant_download "$u" && qdrant_bin_works "$BIN_DIR/qdrant"; then
      echo "  qdrant binary OK ($("$BIN_DIR/qdrant" --version 2>&1 | head -1))"
      return 0
    fi
    echo "  candidate unusable, trying next"
    rm -f "$BIN_DIR/qdrant"
  done

  cat >&2 <<'EOF'
ERROR: could not obtain a runnable qdrant binary.

If the failure mentioned GLIBC, this host's glibc is older than the gnu build requires
and the statically-linked musl build should have been used. Fetch it manually:

  cd .cache/bin
  curl -fLO https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz
  tar xzf qdrant-x86_64-unknown-linux-musl.tar.gz && chmod +x qdrant && ./qdrant --version
EOF
  return 1
}

start_qdrant_native() {
  if qdrant_ready; then
    echo "  qdrant already running on ${QDRANT_PORT}"
    return 0
  fi
  ensure_qdrant_bin
  echo "  starting qdrant on ${QDRANT_PORT} (grpc ${QDRANT_GRPC_PORT})"
  # Qdrant takes all configuration from QDRANT__* env vars, so no config file is needed.
  QDRANT__STORAGE__STORAGE_PATH="$REPO_ROOT/data/qdrant" \
  QDRANT__STORAGE__SNAPSHOTS_PATH="$REPO_ROOT/data/qdrant/snapshots" \
  QDRANT__SERVICE__HTTP_PORT="$QDRANT_PORT" \
  QDRANT__SERVICE__GRPC_PORT="$QDRANT_GRPC_PORT" \
  QDRANT__TELEMETRY_DISABLED=true \
    nohup "$BIN_DIR/qdrant" > "$REPO_ROOT/logs/qdrant.log" 2>&1 &
  echo $! > "$RUN_DIR/qdrant.pid"
}

# ------------------------------------------------------------------ native: redis
ensure_redis_bin() {
  # Prefer a system redis-server; many images already ship one.
  if command -v redis-server >/dev/null 2>&1; then
    printf '%s' "$(command -v redis-server)"
    return 0
  fi
  local built="$REPO_ROOT/.cache/redis-stable/src/redis-server"
  if [[ -x "$built" ]]; then
    printf '%s' "$built"
    return 0
  fi
  echo "  building redis from source (~1 min)" >&2
  command -v make >/dev/null 2>&1 || { echo "ERROR: make required to build redis" >&2; return 1; }
  local tar="$REPO_ROOT/.cache/redis-stable.tar.gz"
  curl -fL --max-time 300 -o "$tar" https://download.redis.io/redis-stable.tar.gz >&2
  tar xzf "$tar" -C "$REPO_ROOT/.cache" >&2
  rm -f "$tar"
  # Only redis-server is needed; building the full suite wastes minutes. Redis' top-level
  # Makefile forwards unknown targets to src/, but that has varied across versions, so
  # fall back to a full build rather than failing on a Makefile quirk.
  make -C "$REPO_ROOT/.cache/redis-stable" -j"$(nproc)" redis-server >&2 \
    || make -C "$REPO_ROOT/.cache/redis-stable" -j"$(nproc)" >&2
  [[ -x "$built" ]] || { echo "ERROR: redis build failed - see output above" >&2; return 1; }
  printf '%s' "$built"
}

start_redis_native() {
  if redis_ready; then
    echo "  redis already running on ${REDIS_PORT}"
    return 0
  fi
  local bin
  bin="$(ensure_redis_bin)"
  echo "  starting redis on ${REDIS_PORT} using $bin"
  # Redis will warn about `vm.overcommit_memory` at startup. IGNORE IT HERE: that warning
  # is about fork-based background saves (RDB/AOF rewrite), and redis.benchmark.conf turns
  # persistence off entirely (`appendonly no`, `save ""`). Nothing forks, so nothing can
  # fail for want of overcommit. Fixing it would need root, which this path exists to avoid.
  # redis.benchmark.conf carries the settings that matter -- above all `databases 64`,
  # without which the runner cannot give each worker its own logical DB and parallelism
  # silently caps at 16. CLI flags after the config file override it, so port and dir are
  # applied on top of the shared config rather than duplicated into it.
  nohup "$bin" "$REPO_ROOT/redis.benchmark.conf" \
      --port "$REDIS_PORT" \
      --dir "$REPO_ROOT/data/redis" \
      --daemonize no \
      > "$REPO_ROOT/logs/redis.log" 2>&1 &
  echo $! > "$RUN_DIR/redis.pid"
}

# ------------------------------------------------------------------ actions
do_start() {
  if docker_usable; then
    echo "--- starting backends via docker compose ---"
    docker compose -f "$COMPOSE_FILE" up -d
    echo "mode: docker"
  else
    echo "--- docker unavailable; starting backends as user processes ---"
    if [[ "${BACKEND_MODE:-auto}" == "native" ]]; then
      echo "  (BACKEND_MODE=native requested)"
    elif ! command -v docker >/dev/null 2>&1; then
      echo "  reason: docker not installed"
    elif ! docker info >/dev/null 2>&1; then
      echo "  reason: docker daemon unreachable or user lacks permission (not in 'docker' group)"
    fi
    start_redis_native
    start_qdrant_native
    echo "mode: native"
  fi

  echo
  echo "--- waiting for readiness ---"
  local rc=0
  wait_ready redis  redis_ready  60 || rc=1
  wait_ready qdrant qdrant_ready 90 || rc=1
  return $rc
}

do_stop() {
  if docker_usable && docker compose -f "$COMPOSE_FILE" ps -q 2>/dev/null | grep -q .; then
    echo "stopping docker stack"
    docker compose -f "$COMPOSE_FILE" down
  fi
  for svc in redis qdrant; do
    local pf="$RUN_DIR/${svc}.pid"
    if [[ -f "$pf" ]]; then
      local pid
      pid="$(cat "$pf")"
      if kill -0 "$pid" 2>/dev/null; then
        echo "stopping $svc (pid $pid)"
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 10); do
          kill -0 "$pid" 2>/dev/null || break
          sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
      fi
      rm -f "$pf"
    fi
  done
  echo "stopped"
}

do_status() {
  if redis_ready;  then echo "redis  : UP   (127.0.0.1:${REDIS_PORT})";  else echo "redis  : DOWN (127.0.0.1:${REDIS_PORT})"; fi
  if qdrant_ready; then echo "qdrant : UP   (127.0.0.1:${QDRANT_PORT})"; else echo "qdrant : DOWN (127.0.0.1:${QDRANT_PORT})"; fi
}

case "$ACTION" in
  start)
    if do_start; then
      echo
      do_status
      echo
      echo "next: python -m benchmarks.preflight"
    else
      echo
      do_status
      echo
      echo "one or more backends failed to start. Check logs/redis.log and logs/qdrant.log." >&2
      exit 1
    fi
    ;;
  stop)   do_stop ;;
  status) do_status ;;
  *)
    echo "usage: $0 [start|stop|status]" >&2
    exit 2
    ;;
esac
