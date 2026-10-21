#!/usr/bin/env bash
# Self-hosted GPU runner without Docker, for a node where the account has no
# Docker access (the DGX Spark "spark": aarch64, one GB10, no sudo).
#
# First time (token: repo Settings -> Actions -> Runners -> New self-hosted runner,
# copy only the --token value):
#   RUNNER_TOKEN=<token> bash native.sh setup
# Then:
#   bash native.sh start     # start in the background (also re-run after a reboot)
#   bash native.sh stop
#   bash native.sh status
#   bash native.sh logs
#
# Jobs run as your user, so the OOM-score problem of the Docker runner on sp-1
# does not apply. Job environment comes from the runner's .env file, written
# by setup: EVOLVETS_DATA (forecast caches, ledgers, venvs, model weights),
# the package and model-weight mirrors, and PATH with uv.
set -euo pipefail
cmd="${1:-status}"
dir="${RUNNER_DIR:-$HOME/actions-runner}"
data="${EVOLVETS_DATA:-$HOME/evolve-data}"
name="${RUNNER_NAME:-spark}"
labels="${RUNNER_LABELS:-gpu,spark}"
url="${RUNNER_URL:-https://github.com/kostagjorgjievski/evolve}"
pidfile="$dir/runner.pid"

running() { [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; }

case "$cmd" in
  setup)
    : "${RUNNER_TOKEN:?setup needs RUNNER_TOKEN}"
    mkdir -p "$dir" "$data"
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh \
        || python3 -m pip install --user --break-system-packages -i https://pypi.tuna.tsinghua.edu.cn/simple uv
    fi
    uv --version
    cd "$dir"
    if [ ! -x ./run.sh ]; then
      case "$(uname -m)" in aarch64) arch=arm64 ;; x86_64) arch=x64 ;; *) echo "unsupported arch"; exit 1 ;; esac
      version="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p')"
      curl -fsSL -o runner.tgz "https://github.com/actions/runner/releases/download/v${version}/actions-runner-linux-${arch}-${version}.tar.gz"
      tar xzf runner.tgz && rm runner.tgz
    fi
    [ -f .runner ] || ./config.sh --unattended --replace --url "$url" --token "$RUNNER_TOKEN" \
      --name "$name" --labels "$labels" --work _work
    # The runner reads this file into every job's environment. Add
    # EVOLVETS_LLM_API_KEY=... here by hand for the Phase 1 pilot (never commit it).
    touch .env && chmod 600 .env
    for kv in "EVOLVETS_DATA=$data" "HF_HOME=$data/hf" "UV_CACHE_DIR=$data/uv-cache" \
              "UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple" "HF_ENDPOINT=https://hf-mirror.com" \
              "PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"; do
      key="${kv%%=*}"
      grep -q "^${key}=" .env && sed -i "s#^${key}=.*#${kv}#" .env || echo "$kv" >> .env
    done
    # Restart the runner after a reboot.
    ( crontab -l 2>/dev/null | grep -v 'native.sh start'; echo "@reboot bash $dir/native.sh start" ) | crontab - || true
    cp "$0" "$dir/native.sh" 2>/dev/null || true
    echo "setup done; now: bash native.sh start"
    ;;
  start)
    cd "$dir"
    if running; then echo "already running (pid $(cat "$pidfile"))"; exit 0; fi
    nohup ./run.sh >>"$dir/runner.log" 2>&1 &
    echo $! > "$pidfile"
    sleep 5; tail -n 5 "$dir/runner.log"
    ;;
  stop)
    if running; then kill "$(cat "$pidfile")" || true; fi
    pkill -f "$dir/bin/Runner.Listener" 2>/dev/null || true
    rm -f "$pidfile"; echo stopped
    ;;
  status)
    if running; then echo "running (pid $(cat "$pidfile"))"; else echo "not running"; fi
    ;;
  logs)
    tail -n 50 "$dir/runner.log"
    ;;
  *) echo "usage: native.sh setup|start|stop|status|logs"; exit 1 ;;
esac
