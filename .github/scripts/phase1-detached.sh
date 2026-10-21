#!/usr/bin/env bash
# W12 (TASK_018): run the Phase 1 pilot detached from the GitHub runner on sp-1.
#
# Same pattern as foundation-detached.sh: started with `docker exec -d`, the
# processes get oom_score_adj 0 and survive runner disconnects. No GPU is used
# (the foundation forecasts are cached). The LLM key comes from the container
# environment and is never written to disk or logged.
#
# Usage (on sp-1; the key is read without echo and passed by name only):
#   read -rs EVOLVETS_LLM_API_KEY && export EVOLVETS_LLM_API_KEY
#   docker exec -d -e EVOLVETS_LLM_API_KEY evolve-gpu-runner bash -c \
#     'curl -fsSL https://raw.githubusercontent.com/kostagjorgjievski/evolve/main/.github/scripts/phase1-detached.sh | bash -s -- <tag>'
# Run ids: phase1-<condition>-<tag>, phase1-predictive-replay-<tag>,
# phase1-holdout-<condition>-<tag>. Status: /data/evolve/phase1-<tag>/status.txt
set -uo pipefail
tag="${1:?tag}"
conditions="${2:-predictive evaluate_all fixed random bandit_ucb bo_gp shuffled weights_only random_param}"
base="${EVOLVETS_DATA:-/data}/evolve"
work="$base/phase1-$tag"
repo=kostagjorgjievski/evolve
mkdir -p "$work/logs"
exec >>"$work/launcher.log" 2>&1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a "$work/status.txt"; }
status "start tag=$tag conditions=$conditions oom_score_adj=$(cat /proc/self/oom_score_adj)"
if [ -z "${EVOLVETS_LLM_API_KEY:-}" ]; then
  status "EVOLVETS_LLM_API_KEY is not set in the container environment; pass it with docker exec -e (see the header)"
  exit 1
fi
rm -rf "$work/src" && mkdir -p "$work/src" && cd "$work/src" || exit 1
if timeout 120 git clone -q --depth 1 "https://github.com/$repo.git" . 2>/dev/null; then
  status "git checkout $(git rev-parse HEAD)"
else
  find . -mindepth 1 -delete
  sha="$(curl -fsSL --retry 5 "https://api.github.com/repos/$repo/commits/main" | "$base/venv-dev/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["sha"])')" || exit 1
  curl -fsSL --retry 5 --max-time 300 "https://api.github.com/repos/$repo/tarball/$sha" | tar -xz --strip-components=1 || exit 1
  git init -q . && git add -A >/dev/null && git -c user.name=ci -c user.email=ci@example.invalid commit -q -m "tarball of $sha"
  status "tarball of main $sha recorded as local commit $(git rev-parse HEAD)"
fi
py="$base/venv-dev/bin/python"
export PYTHONPATH="$work/src/src" CUDA_VISIBLE_DEVICES=""
cli() { "$py" -c 'import sys; from evolvets.runner.cli import main; sys.exit(main())' "$@"; }
paths=(--ledger-root "$base/ledgers" --cache-root "$base/cache_store"
       --foundation-root "$base/foundation_cache" --data-root "$base/data" --manifest-dir data/manifests)
lineage=(--dataset synthetic-shift3 --pool-spec simple+foundation --seed 0
         --proposals-per-generation 3 --eval-slots 1 --max-evaluations 16
         --eval-timeout-seconds 1800 --eval-memory-gib 8)
ran=()
for c in $conditions; do
  id="phase1-$c-$tag"
  status "lineage $c ($id) start"
  cli phase1 lineage --condition "$c" "${lineage[@]}" --run-id "$id" "${paths[@]}" >"$work/logs/lineage-$c.log" 2>&1
  code=$?
  status "lineage $c ($id) exit $code"
  [ "$code" -eq 0 ] && ran+=("$id")
done
if [[ " $conditions " == *" predictive "* ]]; then
  id="phase1-predictive-replay-$tag"
  status "replay of phase1-predictive-$tag ($id) start"
  cli phase1 lineage --condition predictive "${lineage[@]}" --replay-from "phase1-predictive-$tag" \
    --run-id "$id" "${paths[@]}" >"$work/logs/replay-predictive.log" 2>&1
  status "replay ($id) exit $?"
fi
for r in "${ran[@]}"; do
  c="${r#phase1-}"; c="${c%-$tag}"
  id="phase1-holdout-$c-$tag"
  status "holdout of $r ($id) start"
  cli phase1 holdout --from-run "$r" --dataset synthetic-shift3-holdout --pool-spec simple+foundation \
    --run-id "$id" "${paths[@]}" >"$work/logs/holdout-$c.log" 2>&1
  status "holdout of $r ($id) exit $?"
done
status "done"
