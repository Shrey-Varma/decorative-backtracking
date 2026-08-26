"""Persistence, run manifests, determinism helpers, shared record types.

Everything downstream of generation is keyed on *absolute token positions*
into `ChainRecord.token_ids` (prompt + completion), so truncation points and
event spans stay reproducible across backends (§9).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


# ------------------------------------------------------------------- record types


@dataclass
class ChainRecord:
    chain_id: str
    problem_id: str
    dataset: str
    prompt: str
    gold_answer: str
    seed: int
    token_ids: list[int]            # full sequence: prompt + completion
    prompt_token_len: int           # completion starts at this index
    text: str                       # decoded completion text
    token_char_offsets: list[tuple[int, int]]  # per completion token, char span in `text`
    trace_token_span: Optional[tuple[int, int]] = None  # absolute [start, end) of think span
    final_answer: Optional[str] = None
    final_correct: Optional[bool] = None
    generation_config: dict = field(default_factory=dict)

    def completion_token_to_abs(self, i: int) -> int:
        return self.prompt_token_len + i

    def abs_to_completion_token(self, t: int) -> int:
        return t - self.prompt_token_len


@dataclass
class EventRecord:
    chain_id: str
    event_id: str
    onset: int                      # t0, absolute token index
    end: int                        # t1, absolute token index (inclusive)
    marker: str                     # lexicon entry that fired
    marker_text: str                # surface text matched
    context: str                    # short text window for human validation
    source: str = "lexical"         # lexical | feature


@dataclass
class ProbeResult:
    chain_id: str
    position: int                   # absolute truncation point t
    raw: str
    parsed: Optional[str]


@dataclass
class ContinuationRecord:
    chain_id: str
    event_id: str
    condition: str
    seed: int
    truncation: int                 # absolute position the continuation started from
    ablation_mode: Optional[str]
    position_mask: Optional[str]
    feature_ids: list[int]
    text: str
    final_answer: Optional[str]
    outcome: str                    # REVERT | NO_CHANGE | DISRUPTED
    # fingerprint of the full intervention (features, mode, mask, truncation);
    # part of the cache key so config/feature changes never reuse stale runs
    intervention_key: str = ""
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------- json(l) IO


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(obj), indent=2, default=str))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(_to_jsonable(obj), default=str) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_chains(path: str | Path) -> list[ChainRecord]:
    out = []
    for d in read_jsonl(path):
        d["token_char_offsets"] = [tuple(o) for o in d["token_char_offsets"]]
        if d.get("trace_token_span") is not None:
            d["trace_token_span"] = tuple(d["trace_token_span"])
        out.append(ChainRecord(**d))
    return out


def load_events(path: str | Path) -> list[EventRecord]:
    return [EventRecord(**d) for d in read_jsonl(path)]


def group_events_by_chain(events: list[EventRecord]) -> dict[str, list[EventRecord]]:
    """One grouping used by every stage — grouping semantics must not drift
    between feature selection, trajectories, and the necessity test."""
    out: dict[str, list[EventRecord]] = {}
    for ev in events:
        out.setdefault(ev.chain_id, []).append(ev)
    return out


# --------------------------------------------------------------------------- caches


class JsonlCache:
    """Append-only cache keyed on a stable string; used for probes, answers,
    and continuations so interrupted runs resume without recompute (§11 cost)."""

    def __init__(self, path: str | Path, key_fields: list[str]):
        self.path = Path(path)
        self.key_fields = key_fields
        self._index: dict[str, dict] = {}
        for d in read_jsonl(self.path):
            self._index[self._key_of(d)] = d

    def _key_of(self, d: dict) -> str:
        return "\x1f".join(str(d[k]) for k in self.key_fields)

    def key(self, **kwargs: Any) -> str:
        return "\x1f".join(str(kwargs[k]) for k in self.key_fields)

    def get(self, **kwargs: Any) -> Optional[dict]:
        return self._index.get(self.key(**kwargs))

    def put(self, record: Any) -> None:
        d = _to_jsonable(record)
        self._index[self._key_of(d)] = d
        append_jsonl(self.path, d)

    def __len__(self) -> int:
        return len(self._index)

    def values(self) -> list[dict]:
        return list(self._index.values())


# ----------------------------------------------------------------- manifests / runs


def config_hash(config_dict: dict) -> str:
    blob = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def git_sha() -> Optional[str]:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
                cwd=Path(__file__).parent,
            ).stdout.strip()
        )
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for pkg in ("torch", "transformers", "transformer_lens", "sae_lens", "numpy"):
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = "not-installed"
    return versions


def write_manifest(run_dir: str | Path, config_dict: dict, extra: Optional[dict] = None) -> Path:
    """Every script writes one of these into its output dir (§4)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config_dict),
        "git_sha": git_sha(),
        "argv": sys.argv,
        "package_versions": _package_versions(),
        "config": config_dict,
    }
    if extra:
        manifest.update(extra)
    path = run_dir / "manifest.json"
    write_json(path, manifest)
    return path


def resolve_run_dir(cfg_dict: dict, run_name: Optional[str], output_dir: str) -> Path:
    name = run_name or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{config_hash(cfg_dict)}"
    return Path(output_dir) / name


# ---------------------------------------------------------------------- determinism


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_hash_fraction(key: str, salt: int = 0) -> float:
    """Deterministic [0,1) hash used for pilot/test chain splits (§10)."""
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64
