"""Answer extraction + verification (build order step 4, §8).

Rule-based first (boxed / final-answer parsing with numeric + string
normalization); optional LLM verifier for semantic equivalence, off by
default. Verification results cache by (problem_id, answer_text).
"""

from __future__ import annotations

import logging
import re
from fractions import Fraction
from typing import Optional

from .config import VerifierConfig
from .io_utils import JsonlCache

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------- extraction


def _balanced_prefix(text: str) -> tuple[str, bool]:
    """Consume `text` as the inside of an already-open brace: return the
    content up to the brace that balances it, and whether it ever balanced."""
    depth = 1
    out = []
    for c in text:
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out), True
        out.append(c)
    return "".join(out), False


def extract_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...}, with balanced-brace matching. A
    truncated (never-balanced) box is unparseable — returning the partial
    content would hand downstream classifiers a garbage answer."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    content, closed = _balanced_prefix(text[start + len("\\boxed{"):])
    return content if closed else None


_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|the answer)(?:\s+is)?\s*[:=]?\s*(.+?)(?:\.\s|\n|$)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_explicit_answer(text: str) -> Optional[str]:
    """Boxed or explicit answer-cue only — NO bare-number fallback. This is
    what outcome classification must use: a continuation that hits the token
    cap mid-reasoning re-mentions numbers constantly, and scoring its last
    stray number as 'the final answer' turns truncation into spurious REVERTs."""
    boxed = extract_boxed(text)
    if boxed is not None and boxed.strip():
        return boxed.strip()
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if matches:
        cand = matches[-1].group(1).strip()
        if cand:
            return cand
    return None


def extract_answer(text: str) -> Optional[str]:
    """Rule-based final-answer extraction: boxed, then answer-cue line, then
    last bare number. The number fallback makes this suitable for chain
    filtering (§9), NOT for outcome classification — use
    extract_explicit_answer there."""
    explicit = extract_explicit_answer(text)
    if explicit is not None:
        return explicit
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1]
    return None


def extract_probe_answer(raw: str, stop_string: Optional[str] = "}") -> Optional[str]:
    """Parse the force-answer probe output. With stop_string '}', the probe
    template ends in an open '\\boxed{', so the answer runs to the brace that
    *balances* it (answers like \\frac{1}{2} contain internal braces); if the
    bounded probe span never closes it, take everything decoded so far — the
    probe span is short and template-constrained, unlike free continuations.
    Any other stop_string is a plain delimiter."""
    if stop_string == "}":
        raw, _closed = _balanced_prefix(raw)
    elif stop_string and stop_string in raw:
        raw = raw[: raw.index(stop_string)]
    ans = raw.strip().splitlines()[0].strip() if raw.strip() else None
    return normalize_answer(ans) if ans else None


# -------------------------------------------------------------------- normalization


_LATEX_STRIP = [
    (re.compile(r"\\left\s*"), ""),
    (re.compile(r"\\right\s*"), ""),
    (re.compile(r"\\dfrac"), r"\\frac"),
    (re.compile(r"\\tfrac"), r"\\frac"),
    (re.compile(r"\\!|\\,|\\;|\\:"), ""),
    (re.compile(r"\\text\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\mathrm\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\\\$"), ""),
]


# A space carries meaning only BETWEEN two alphanumerics ("no solution"); a
# space touching punctuation, an operator, or a LaTeX backslash is pure
# formatting. Collapsing only the latter makes `\left( 3, \frac{\pi}{2} \right)`
# and `(3, \dfrac{\pi}{2})` compare equal without merging distinct word
# answers. Applied twice so runs like "( 3 , 1 )" fully close up.
_FORMATTING_SPACE = re.compile(r"(?<![0-9A-Za-z]) +|(?<= ) +| +(?![0-9A-Za-z])")


def normalize_answer(ans: str) -> str:
    s = ans.strip()
    s = s.strip("$").strip()
    for pat, repl in _LATEX_STRIP:
        s = pat.sub(repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(2):
        s = _FORMATTING_SPACE.sub("", s)
    s = s.rstrip(".").strip()
    # strip thousands separators inside numbers: 1,234,567 -> 1234567
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # degree / unit fluff commonly appended by models
    s = re.sub(r"(\^\{?\\circ\}?|°)$", "", s).strip()
    return s


def _try_numeric(s: str) -> Optional[Fraction]:
    s = s.strip()
    frac = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if frac:
        try:
            return Fraction(int(frac.group(1)), int(frac.group(2)))
        except (ValueError, ZeroDivisionError):
            return None
    ratio = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if ratio:
        try:
            return Fraction(int(ratio.group(1)), int(ratio.group(2)))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return Fraction(str(float(s)))
    except (ValueError, OverflowError):
        return None


def answers_equal(a: Optional[str], b: Optional[str]) -> bool:
    """Equality up to normalization; numeric answers compare as exact rationals."""
    if a is None or b is None:
        return False
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    fa, fb = _try_numeric(na), _try_numeric(nb)
    if fa is not None and fb is not None:
        return fa == fb
    return na.lower() == nb.lower()


# -------------------------------------------------------------------- verification


class LLMVerifier:
    """Optional semantic-equivalence verifier against an OpenAI-compatible
    endpoint (§8). Only consulted when rule-based equality says False."""

    PROMPT = (
        "You are grading a math answer. Reply with exactly YES or NO.\n"
        "Question-independent check: are these two final answers mathematically "
        "equivalent?\nGold answer: {gold}\nModel answer: {pred}\n"
    )

    def __init__(self, cfg: VerifierConfig):
        import os
        from openai import OpenAI

        self.model = cfg.model
        self.client = OpenAI(
            base_url=cfg.base_url, api_key=os.environ.get(cfg.api_key_env, "unset")
        )

    def equivalent(self, pred: str, gold: str) -> bool:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self.PROMPT.format(gold=gold, pred=pred)}],
            max_tokens=4,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip().upper().startswith("YES")


class AnswerVerifier:
    """verify(problem_id, answer_text, gold) -> {raw, parsed, correct}, cached
    by (problem_id, answer_text)."""

    def __init__(self, cache_path, verifier_cfg: Optional[VerifierConfig] = None):
        self.cache = JsonlCache(cache_path, key_fields=["problem_id", "answer_text"])
        self.llm: Optional[LLMVerifier] = None
        if verifier_cfg and verifier_cfg.enabled:
            self.llm = LLMVerifier(verifier_cfg)

    def verify(self, problem_id: str, answer_text: str, gold: str) -> dict:
        hit = self.cache.get(problem_id=problem_id, answer_text=answer_text)
        if hit is not None:
            return hit
        parsed = extract_answer(answer_text) if answer_text else None
        correct = answers_equal(parsed, gold) if parsed is not None else False
        if not correct and parsed is not None and self.llm is not None:
            try:
                correct = self.llm.equivalent(parsed, gold)
            except Exception as e:
                logger.warning("LLM verifier failed for %s: %s", problem_id, e)
        record = {
            "problem_id": problem_id,
            "answer_text": answer_text,
            "raw": answer_text,
            "parsed": parsed,
            "correct": bool(correct),
        }
        self.cache.put(record)
        return record
