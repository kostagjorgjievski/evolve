#!/usr/bin/env bash
# Run this ON the cluster GPU node (over SSH) before installing the GitHub
# runner. It changes nothing; it only reports whether the node is ready.
#   bash gpu-runner-preflight.sh
set -uo pipefail

ok()   { printf '  [ok]   %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=1; }
warn() { printf '  [warn] %s\n' "$1"; }
fail=0

echo "Host: $(hostname)  OS: $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")  Arch: $(uname -m)"

echo "1. GPUs"
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "nvidia-smi found"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | wc -l)
  echo "     GPUs with running processes right now: ${apps}"
else
  bad "nvidia-smi not found (NVIDIA driver missing?)"
fi

echo "2. Outbound access the runner needs"
for url in https://github.com https://api.github.com https://pipelines.actions.githubusercontent.com \
           https://objects.githubusercontent.com https://pypi.org; do
  code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$url")
  if [ "$code" != "000" ]; then ok "$url -> HTTP $code"; else bad "$url unreachable (needs a proxy?)"; fi
done
[ -n "${https_proxy:-${HTTPS_PROXY:-}}" ] && warn "a proxy is set in this shell: ${https_proxy:-$HTTPS_PROXY}"

echo "3. Scheduler"
if command -v sbatch >/dev/null 2>&1; then
  warn "Slurm found (sbatch). GPU jobs should go through Slurm; tell Claude before installing."
else
  ok "no Slurm; jobs will use GPUs directly after the free-GPU check"
fi

echo "4. Tools"
for t in git curl tar; do command -v "$t" >/dev/null && ok "$t" || bad "$t missing"; done
df -h "$HOME" | tail -1 | awk '{print "     free disk in $HOME: " $4}'

echo
if [ "$fail" -eq 0 ]; then echo "READY: install the runner."; else echo "NOT READY: send this output to Claude."; fi
