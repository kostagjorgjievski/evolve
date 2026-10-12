#!/usr/bin/env bash
# Run a command, stream its output, and on failure publish the last lines as an
# error annotation (readable through the API even when the job log is not).
# Usage: run-logged.sh <title> <command> [args...]
set -uo pipefail
title="$1"; shift
log="$(mktemp)"
"$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  msg="$(printf 'exit %s: %s\n' "$rc" "$*"; tail -n 15 "$log")"
  msg="${msg//'%'/'%25'}"; msg="${msg//$'\r'/'%0D'}"; msg="${msg//$'\n'/'%0A'}"
  echo "::error title=${title}::${msg}"
fi
rm -f "$log"
exit "$rc"
