# Running the agent loop

The developer, reviewer and planner agents run in GitHub Actions and hand work to each other through PRs, comments and issues. You only step in when they open a `needs-human` issue.

## How work flows

| Event | Agent woken | What it does |
|---|---|---|
| A PR merges into `main` | Developer | Starts the next PR of the active task, or opens `[agent:needs-plan]` when the task is finished |
| CI passes on a PR | Reviewer | Reviews and probes; squash-merges, or comments "changes requested" |
| CI fails on a PR | Developer | Fixes it |
| Reviewer comments "changes requested" | Developer | Pushes fixes and replies item by item |
| `[agent:needs-plan]` issue opened | Planner | Writes the next task spec as a PR, or escalates to you at a gate |
| Every 2 hours | Developer and reviewer | Safety net: picks up anything a missed event left waiting |

## One-time setup

1. Install the Claude GitHub App on this repository.
2. Add repository secrets (Settings → Secrets and variables → Actions):
   - `CLAUDE_CODE_OAUTH_TOKEN`: from `claude setup-token` (uses your Claude subscription). Used by the reviewer and planner.
   - `ZAI_API_KEY`: your Z.ai API key. The developer agent runs GLM through Z.ai's Anthropic-compatible endpoint.
   - `AGENTS_GH_TOKEN`: a fine-grained personal access token limited to this repository, with Contents, Pull requests and Issues set to read and write, and Actions and Metadata set to read. Leave Workflows off, so agents cannot push changes to these workflow files.
3. Optional repository variables (same settings page, Variables tab): `AGENTS_DAILY_RUN_CAP` (default 40), `AGENTS_PAUSED` (`true` to stop everything), `DEV_MODEL` (the developer's Z.ai model id, default `glm-5.3`).
4. Create two issue labels: `needs-human` and `fyi-human`.
5. Branch protection on `main`: require a pull request and the `checks` status check; required approvals 0 (the agents act through your account, which cannot approve its own PRs).

## Controls

- **Pause everything:** set the variable `AGENTS_PAUSED` to `true`. Unset it, or set it to `false`, to resume.
- **Kick a run manually:** Actions tab → pick the agent workflow → Run workflow.
- **Answer an escalation:** read the `needs-human` issue, reply with your decision (or merge, tag or approve whatever it asks), then close it. Closing the last open `needs-human` issue resumes the agents at their next trigger; run a workflow manually to resume immediately.
- **Watch:** the Actions tab shows every run; PR threads show every review and reply.

## When you'll be needed

- Research gate decisions (for example Gate 1: harden the generator, or proceed).
- New heavy dependencies, GPUs or paid services.
- Protocol changes to frozen components, and tagging frozen versions (`fyi-human`).
- The start of each new research phase.
- A PR stuck after four review rounds.
- Any PR touching `AGENTS.md`, `docs/agents/`, `.github/` or `.importlinter`. Agents never merge these; you do.
