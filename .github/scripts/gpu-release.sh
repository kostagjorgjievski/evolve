#!/usr/bin/env bash
# Release GPU locks taken by gpu-pick.sh. Safe to run when nothing was picked.
set -uo pipefail
for lock in ${EVOLVETS_GPU_LOCKS:-}; do
  case "$lock" in
    */evolvets-gpu-*.lock) rm -rf "$lock" && echo "Released ${lock}" ;;
  esac
done
exit 0
