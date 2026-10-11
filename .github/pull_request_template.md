## What and why

## How it was tested

## Part of a stack?
(Which PR number in the stack, what comes before and after.)

## Integrity checklist
- [ ] No agent-editable package can reach sealed or live data (import-linter passes)
- [ ] No look-ahead: targets only reach code through arena label release
- [ ] No frozen component changed (or this is a `protocol/` PR with version bump)
- [ ] All randomness is seeded from config
- [ ] No data, cache, ledger, secret or large file committed
- [ ] No test weakened, skipped or deleted to make CI pass

## Notes for the reviewer
