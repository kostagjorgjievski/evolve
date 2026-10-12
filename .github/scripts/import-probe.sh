#!/usr/bin/env bash
# Prove an import contract rejects a planted violation.
# Usage: import-probe.sh "<import statement>" "<contract name>"
# Passes only if lint-imports exits non-zero AND reports that contract as BROKEN;
# any other failure (crash, missing config) fails the probe.
set -euo pipefail
statement="$1"
contract="$2"
probe=src/evolvets/updater/_violation_probe.py
echo "$statement" > "$probe"
set +e
out=$(uv run lint-imports 2>&1)
status=$?
set -e
rm "$probe"
if [ "$status" -eq 0 ]; then
  echo "lint-imports accepted a forbidden import: $statement"
  printf '%s\n' "$out"
  exit 1
fi
if ! printf '%s\n' "$out" | grep -q "${contract}.*BROKEN"; then
  echo "lint-imports exited $status without reporting '$contract' as BROKEN"
  printf '%s\n' "$out"
  exit 1
fi
printf '%s\n' "$out" | grep "${contract}.*BROKEN"
echo "OK: '$contract' rejects: $statement"
