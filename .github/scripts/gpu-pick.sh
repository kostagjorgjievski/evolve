#!/usr/bin/env bash
# Pick idle GPUs on a shared node before running anything on them.
#
# A GPU counts as free only if ALL of these hold:
#   - no compute process is running on it (nvidia-smi --query-compute-apps);
#   - memory used is below GPU_FREE_MEM_MIB (default 1024 MiB);
#   - utilization is below GPU_FREE_UTIL_PCT (default 10%);
#   - no other EvolveTS job holds its lock (/tmp/evolvets-gpu-<index>.lock).
#
# Usage: gpu-pick.sh [count]   (default 1)
# Waits up to GPU_WAIT_MINUTES (default 30) for enough free GPUs, checking
# every 60 s. On success, writes CUDA_VISIBLE_DEVICES and EVOLVETS_GPU_LOCKS
# to $GITHUB_ENV (or prints them when run by hand). Release the locks with
# gpu-release.sh in an `if: always()` step. If not enough GPUs free up in
# time, exits 1 without touching any GPU.
set -euo pipefail

need="${1:-1}"
mem_max="${GPU_FREE_MEM_MIB:-1024}"
util_max="${GPU_FREE_UTIL_PCT:-10}"
wait_min="${GPU_WAIT_MINUTES:-30}"
lock_dir="${GPU_LOCK_DIR:-/tmp}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::error title=GPU pick::nvidia-smi not found on $(hostname)"
  exit 1
fi

busy_uuids() {
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u
}

try_pick() {
  local busy picked=()
  busy="$(busy_uuids)"
  while IFS=, read -r idx uuid mem util; do
    idx="${idx// /}"; uuid="${uuid// /}"; mem="${mem// /}"; util="${util// /}"
    if grep -qx "$uuid" <<<"$busy"; then continue; fi
    if [ "$mem" -ge "$mem_max" ] || [ "$util" -ge "$util_max" ]; then continue; fi
    # mkdir is atomic: only one job can create the lock directory.
    if mkdir "${lock_dir}/evolvets-gpu-${idx}.lock" 2>/dev/null; then
      echo "$$ $(date -u +%FT%TZ) ${GITHUB_RUN_ID:-manual}" > "${lock_dir}/evolvets-gpu-${idx}.lock/owner"
      picked+=("$idx")
      [ "${#picked[@]}" -ge "$need" ] && break
    fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
            --format=csv,noheader,nounits)

  if [ "${#picked[@]}" -ge "$need" ]; then
    local ids locks
    ids="$(IFS=,; echo "${picked[*]}")"
    locks="$(for i in "${picked[@]}"; do printf '%s ' "${lock_dir}/evolvets-gpu-${i}.lock"; done)"
    if [ -n "${GITHUB_ENV:-}" ]; then
      echo "CUDA_VISIBLE_DEVICES=${ids}" >> "$GITHUB_ENV"
      echo "EVOLVETS_GPU_LOCKS=${locks}" >> "$GITHUB_ENV"
    else
      echo "export CUDA_VISIBLE_DEVICES=${ids}"
      echo "export EVOLVETS_GPU_LOCKS='${locks}'"
    fi
    echo "Picked free GPU(s) ${ids} on $(hostname)"
    return 0
  fi
  # Not enough: release whatever we grabbed this round and retry later.
  for i in "${picked[@]}"; do rm -rf "${lock_dir}/evolvets-gpu-${i}.lock"; done
  return 1
}

deadline=$(( $(date +%s) + wait_min * 60 ))
while true; do
  if try_pick; then exit 0; fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "::error title=GPU pick::fewer than ${need} free GPU(s) on $(hostname) after ${wait_min} min"
    nvidia-smi
    exit 1
  fi
  echo "Waiting for ${need} free GPU(s) on $(hostname)..."
  sleep 60
done
