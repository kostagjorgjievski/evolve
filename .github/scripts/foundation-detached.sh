#!/usr/bin/env bash
# Build foundation-model forecast caches detached from the GitHub runner.
#
# Why: on sp-1 the runner gives job processes oom_score_adj 500, so when another
# user's job exhausts RAM the kernel kills our build (and often the runner) first;
# the holdout builds lost the runner twice on 2026-10-20. Started with
# `docker exec -d` from the host, these processes get the container default (0)
# and survive runner disconnects (same approach as headroom-detached.sh).
#
# Usage (on sp-1):
#   docker exec -d evolve-gpu-runner bash -c \
#     'curl -fsSL https://raw.githubusercontent.com/kostagjorgjievski/evolve/main/.github/scripts/foundation-detached.sh | bash -s -- <tag> ["datasets"] ["models"] [per_family]'
# Defaults: datasets "synthetic-shift3-holdout synthetic-shift2-holdout",
# models "chronos2 tirex timesfm25", per_family 20. Builds run one after another
# on one free GPU each (gpu-pick.sh); a build resumes from its partial directory.
# Status: /data/evolve/fm-<tag>/status.txt
set -uo pipefail
tag="${1:?tag}"
datasets="${2:-synthetic-shift3-holdout synthetic-shift2-holdout}"
models="${3:-chronos2 tirex timesfm25}"
per_family="${4:-20}"
base="${EVOLVETS_DATA:-/data}/evolve"
work="$base/fm-$tag"
repo=kostagjorgjievski/evolve
mkdir -p "$work/logs"
exec >>"$work/launcher.log" 2>&1
echo "== $(date -u) start tag=$tag datasets=$datasets models=$models per_family=$per_family oom_score_adj=$(cat /proc/self/oom_score_adj)"
rm -rf "$work/src" && mkdir -p "$work/src" && cd "$work/src" || exit 1
if timeout 120 git clone -q --depth 1 "https://github.com/$repo.git" . 2>/dev/null; then
  echo "git checkout $(git rev-parse HEAD)"
else
  find . -mindepth 1 -delete
  curl -fsSL --retry 5 --max-time 300 "https://api.github.com/repos/$repo/tarball/main" | tar -xz --strip-components=1 || exit 1
  echo "tarball of main"
fi
export PYTHONPATH="$work/src/src"
for d in $datasets; do
  for m in $models; do
    case "$m" in
      chronos2) group=fm-chronos ;; tirex) group=fm-tirex ;; toto) group=fm-toto ;;
      moirai2) group=fm-moirai ;; timesfm25) group=fm-timesfm ;;
      *) echo "$d $m unknown model" | tee -a "$work/status.txt"; continue ;;
    esac
    py="$base/venv-$group/bin/python"
    if [ ! -x "$py" ]; then echo "$d $m no venv at $py (run the workflow once first)" | tee -a "$work/status.txt"; continue; fi
    cli() { "$py" -c 'import sys; from evolvets.runner.cli import main; sys.exit(main())' "$@"; }
    log="$work/logs/$d-$m.log"
    unset CUDA_VISIBLE_DEVICES EVOLVETS_GPU_LOCKS
    if ! picked="$(GITHUB_ENV='' GPU_WAIT_MINUTES=120 bash .github/scripts/gpu-pick.sh 1)"; then
      echo "$d $m no free GPU" | tee -a "$work/status.txt"; continue
    fi
    eval "$(grep '^export ' <<<"$picked")"
    echo "$(date -u +%FT%TZ) $d $m GPU $CUDA_VISIBLE_DEVICES" | tee -a "$work/status.txt"
    cli cache build "$d" --per-family "$per_family" --cache-root "$base/cache_store" --data-root "$base/data" >"$log" 2>&1
    code=$?
    attempt=1
    while [ "$code" -eq 0 ]; do
      cli foundation build --dataset "$d" --model "$m" --device cuda --batch-size 32 --per-family "$per_family" \
        --root "$base/foundation_cache" --cache-root "$base/cache_store" --data-root "$base/data" >>"$log" 2>&1
      code=$?
      [ "$code" -eq 0 ] && break
      if [ "$code" -ne 137 ] || [ "$attempt" -ge 6 ]; then break; fi
      echo "$(date -u +%FT%TZ) $d $m SIGKILL on attempt $attempt; resuming in 5 min" | tee -a "$work/status.txt"
      attempt=$((attempt + 1)); code=0; sleep 300
    done
    if [ "$code" -eq 0 ]; then
      cli foundation verify --dataset "$d" --model "$m" --root "$base/foundation_cache" >>"$log" 2>&1
      code=$?
    fi
    bash .github/scripts/gpu-release.sh >/dev/null
    echo "$(date -u +%FT%TZ) $d $m exit $code" | tee -a "$work/status.txt"
  done
done
echo "== $(date -u) done" | tee -a "$work/status.txt"
