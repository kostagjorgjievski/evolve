"""The run lifecycle: one recorded experiment, from run_start to run_end.

A :class:`Run` owns one run's main stream for its lifetime. It writes the
``run_start`` (config blob, protocol versions, clean git commit, budget
limits), appends records while summing what they charge, and guarantees a
``run_end`` — including when the run dies mid-flight, so a run that started
is always visible in the ledger; a failed final write is never silent
(AGENTS.md 2.8).
"""

from __future__ import annotations

import re
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from evolvets.arena.versions import protocol_versions
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.canonical import canonical_bytes
from evolvets.ledger.payloads import Payload, RunEnd, RunStart
from evolvets.ledger.schema import Budget, Record
from evolvets.ledger.store import LedgerWriter

#: Budget fields summed into :attr:`Run.spent`, in schema order.
_BUDGET_FIELDS = (
    "llm_input_tokens",
    "llm_output_tokens",
    "wall_seconds",
    "gpu_seconds",
    "evaluations",
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PREFIX_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class RunError(Exception):
    """Raised when a run cannot start, or is used against its lifecycle."""


def _git(repo: Path, *arguments: str) -> str:
    """Run one ``git -C <repo>`` command; ``RunError`` on any failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RunError(f"cannot run git -C {repo} {' '.join(arguments)}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RunError(f"git -C {repo} {' '.join(arguments)} failed: {detail}")
    return result.stdout


def current_git_commit(repo: Path) -> str:
    """Return the clean ``HEAD`` commit of the repository at ``repo``.

    A missing ``git`` binary, a non-repository or a repository with no commit
    raises :class:`RunError`, and so does any modification or staged change
    to a tracked file: a run must never record a dirty commit, because every
    number must be reproducible from the ledger plus a commit. Untracked
    files are allowed (ledgers, caches and data are untracked or ignored).
    """
    commit = _git(repo, "rev-parse", "HEAD").strip()
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise RunError("working tree has uncommitted changes; commit before starting a run")
    if not _COMMIT_RE.fullmatch(commit):
        raise RunError(f"HEAD is not a 40-lowercase-hex commit id: {commit!r}")
    return commit


def new_run_id(prefix: str) -> str:
    """Return a fresh run id: ``<prefix>-<UTC stamp>-<8 hex>``."""
    if not _PREFIX_RE.fullmatch(prefix):
        raise RunError(f"run id prefix must match ^[a-z0-9][a-z0-9-]*$, got {prefix!r}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


class Run:
    """One recorded experiment; see the module docstring."""

    def __init__(
        self,
        writer: LedgerWriter,
        blobs: BlobStore,
        run_id: str,
        run_start_id: str,
        limits: Budget,
    ) -> None:
        self._writer = writer
        self._blobs = blobs
        self._run_id = run_id
        self._run_start_id = run_start_id
        self._limits = limits
        self._spent = {field: 0.0 for field in _BUDGET_FIELDS}
        self._ended = False

    @classmethod
    def start(
        cls,
        ledger_root: Path,
        run_id: str,
        *,
        condition: str,
        seed: int,
        config: Mapping[str, object],
        budget_limits: Budget,
        git_commit: str,
        requires_predictions: bool = False,
    ) -> Run:
        """Open a run and write its ``run_start``; nothing is written on refusal."""
        if not _RUN_ID_RE.fullmatch(run_id):
            raise RunError(f"run id must match ^[A-Za-z0-9][A-Za-z0-9._-]*$, got {run_id!r}")
        if ".." in run_id:
            raise RunError(f"run id must not contain '..': {run_id!r}")
        try:
            config_bytes = canonical_bytes(config)
        except (TypeError, ValueError) as exc:
            raise RunError(f"config is not canonically serializable: {exc}") from exc
        if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
            raise RunError("config must be a dict with string keys")
        root = Path(ledger_root)
        for stream in ("main", "sealed"):
            if (root / run_id / f"{stream}.jsonl").exists():
                raise RunError(
                    f"refusing to reuse run id {run_id!r}: "
                    f"{root / run_id / (stream + '.jsonl')} already exists"
                )
        writer = LedgerWriter(root, run_id, "main")
        blobs = BlobStore(root / run_id / "blobs")
        try:
            record = writer.append(
                "run_start",
                RunStart(
                    condition=condition,
                    seed=seed,
                    requires_predictions=requires_predictions,
                    config_blob=blobs.put(config_bytes),
                    git_commit=git_commit,
                    protocol_versions=protocol_versions(),
                    budget_limits=budget_limits,
                ),
            )
        except BaseException:
            writer.close()
            raise
        return cls(writer, blobs, run_id, record.record_id, budget_limits)

    @property
    def run_id(self) -> str:
        """The run's id, which is also its ledger directory name."""
        return self._run_id

    @property
    def run_start_id(self) -> str:
        """The record id of the run's ``run_start``."""
        return self._run_start_id

    @property
    def limits(self) -> Budget:
        """The budget limits the run declared at start."""
        return self._limits

    @property
    def spent(self) -> Budget:
        """The field-wise sum of the budgets of every record this Run appended."""
        return Budget(**self._spent)

    @property
    def blobs(self) -> BlobStore:
        """The run's blob store, for readers that must resolve blob references (the improver)."""
        return self._blobs

    def put_blob(self, obj: object) -> str:
        """Store ``obj`` as canonical bytes in the run's blob store."""
        return self._blobs.put(canonical_bytes(obj))

    def append(
        self,
        kind: str,
        payload: Payload,
        *,
        refers_to: Sequence[str] = (),
        budget: Budget | None = None,
    ) -> Record:
        """Append one record and charge its budget to :attr:`spent`."""
        if self._ended:
            raise RunError("the run has ended; no more records can be appended")
        if kind in ("run_start", "run_end"):
            raise RunError(f"append refuses the {kind!r} kind; use Run.start/Run.end")
        record = self._writer.append(kind, payload, refers_to=refers_to, budget=budget)
        for field in _BUDGET_FIELDS:
            self._spent[field] += getattr(record.budget, field)
        return record

    def end(self, status: str, summary: Mapping[str, object]) -> Record:
        """Write the ``run_end`` and close the writer; only once per run."""
        if self._ended:
            raise RunError("the run has already ended")
        record = self._writer.append("run_end", RunEnd(status=status, summary=dict(summary)))
        self._ended = True
        self._writer.close()
        return record

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        """Guarantee a ``run_end``: aborted on any exit that did not end the run.

        If that final write itself fails, the writer is closed and this Run can
        append nothing more. On a normal exit the write error propagates (it is
        never swallowed); when an exception is already in flight, that original
        exception propagates instead — the write error never displaces it.
        """
        if self._ended:
            return
        error = (
            f"{type(exc).__name__}: {exc}" if exc_type is not None else "run exited without run_end"
        )
        try:
            self._writer.append("run_end", RunEnd(status="aborted", summary={"error": error}))
        except BaseException:
            self._ended = True
            self._writer.close()
            if exc_type is not None:
                return  # the original exception propagates; never displace it
            raise  # nothing is in flight; never swallow the write error
        self._ended = True
        self._writer.close()
