#!/usr/bin/env bash
# Run the TASK_015 headroom scan tune (W9's work) detached from the GitHub runner.
#
# Why: on sp-1 the runner gives job processes oom_score_adj 500, so when another
# user's job exhausts RAM the kernel kills our tune (and the runner) first; W9 lost
# the runner twice. Started with `docker exec -d` from the host, these processes get
# the container default (0) and survive runner disconnects.
#
# Usage (on sp-1):
#   docker exec -d evolve-gpu-runner bash -c \
#     'curl -fsSL https://raw.githubusercontent.com/kostagjorgjievski/evolve/main/.github/scripts/headroom-detached.sh | bash -s -- <tag> [datasets...]'
# Run ids: gate1-scan-<dataset>-<tag>. Status: /data/evolve/scan-<tag>/status.txt
set -uo pipefail
tag="${1:?tag}"; shift
datasets="${*:-synthetic-shift1 synthetic-shift2 synthetic-shift3 synthetic-wide}"
base="${EVOLVETS_DATA:-/data}/evolve"
work="$base/scan-$tag"
repo=kostagjorgjievski/evolve
mkdir -p "$work/logs"
exec >>"$work/launcher.log" 2>&1
echo "== $(date -u) start tag=$tag datasets=$datasets oom_score_adj=$(cat /proc/self/oom_score_adj)"
rm -rf "$work/src" && mkdir -p "$work/src" && cd "$work/src" || exit 1
if timeout 120 git clone -q --depth 1 "https://github.com/$repo.git" . 2>/dev/null; then
  echo "git checkout $(git rev-parse HEAD)"
else
  find . -mindepth 1 -delete
  sha="$(curl -fsSL --retry 5 "https://api.github.com/repos/$repo/commits/main" | "$base/venv-dev/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["sha"])')" || exit 1
  curl -fsSL --retry 5 --max-time 300 "https://api.github.com/repos/$repo/tarball/$sha" | tar -xz --strip-components=1 || exit 1
  git init -q . && git add -A >/dev/null && git -c user.name=ci -c user.email=ci@example.invalid commit -q -m "tarball of $sha"
  echo "tarball of main $sha recorded as local commit $(git rev-parse HEAD)"
fi
py="$base/venv-dev/bin/python"
export PYTHONPATH="$work/src/src" CUDA_VISIBLE_DEVICES=""
for d in $datasets; do
  id="gate1-scan-$d-$tag"
  nohup "$py" -c 'import sys; from evolvets.runner.cli import main; sys.exit(main())' gate1 tune \
    --dataset "$d" --pool-spec simple --pool-spec simple+foundation --run-id "$id" \
    --max-wall-seconds 172800 --seed 0 \
    --ledger-root "$base/ledgers" --cache-root "$base/cache_store" \
    --foundation-root "$base/foundation_cache" --data-root "$base/data" \
    --manifest-dir data/manifests > "$work/logs/$d.log" 2>&1 &
  echo "$d $id pid $!" | tee -a "$work/status.txt"
done
wait
for d in $datasets; do echo "$d: $(head -n1 "$work/logs/$d.log")" | tee -a "$work/status.txt"; done
echo "== $(date -u) done" | tee -a "$work/status.txt"
