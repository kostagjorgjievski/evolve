#!/usr/bin/env bash
# Register once (state is kept in the mounted volume), then run forever.
set -euo pipefail
cd /home/runner/actions-runner
if [ ! -f .runner ]; then
  : "${RUNNER_TOKEN:?first start needs RUNNER_TOKEN (Settings -> Actions -> Runners -> New self-hosted runner)}"
  ./config.sh --unattended --replace \
    --url "${RUNNER_URL:-https://github.com/kostagjorgjievski/evolve}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME:-sp-1}" \
    --labels "${RUNNER_LABELS:-gpu}" \
    --work _work
fi
exec ./run.sh
