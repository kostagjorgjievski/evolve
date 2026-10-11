#!/usr/bin/env bash
# Decide whether an agent run may proceed. Writes proceed=true|false to
# $GITHUB_OUTPUT. The pause switch (repo variable AGENTS_PAUSED) is checked in
# the workflow itself; this script handles the remaining stop conditions:
#   1. any open issue labelled `needs-human` pauses every agent;
#   2. a daily cap on event-triggered agent runs (AGENTS_DAILY_RUN_CAP, default 40).
set -euo pipefail

repo="${GITHUB_REPOSITORY:?}"
cap="${AGENTS_DAILY_RUN_CAP:-40}"

fail() {
  echo "::error title=Agent guard::$1"
  exit 1
}

if [ -z "${GH_TOKEN:-}" ]; then
  fail "GH_TOKEN is empty: the AGENTS_GH_TOKEN repository secret is missing or not visible to this workflow."
fi

stop() {
  echo "proceed=false" >> "$GITHUB_OUTPUT"
  echo "::notice title=Agent paused::$1"
  exit 0
}

if ! open_human=$(gh api "repos/${repo}/issues?labels=needs-human&state=open&per_page=100" --jq 'length' 2>&1); then
  fail "Could not list issues with AGENTS_GH_TOKEN (needs Issues: read, scoped to ${repo}): ${open_human}"
fi
if [ "${open_human}" -gt 0 ]; then
  stop "${open_human} open needs-human issue(s); waiting for a human decision."
fi

today=$(date -u +%Y-%m-%d)
runs=0
for wf in agent-developer.yml agent-reviewer.yml agent-planner.yml; do
  n=$(gh api "repos/${repo}/actions/workflows/${wf}/runs?created=>=${today}&per_page=100" \
        --jq '[.workflow_runs[] | select(.event != "schedule")] | length' 2>/dev/null || echo 0)
  runs=$((runs + n))
done
if [ "${runs}" -gt "${cap}" ]; then
  stop "Daily cap reached: ${runs} event-triggered agent runs today (cap ${cap})."
fi

echo "proceed=true" >> "$GITHUB_OUTPUT"
