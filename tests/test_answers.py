from src.answers import (
    AnswerVerifier,
    answers_equal,
    extract_answer,
    extract_boxed,
    extract_explicit_answer,
    extract_probe_answer,
    normalize_answer,
)


def test_extract_boxed_simple():
    assert extract_boxed(r"blah \boxed{42} done") == "42"


def test_extract_boxed_nested_braces():
    assert extract_boxed(r"so \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_boxed_takes_last():
    assert extract_boxed(r"\boxed{1} then \boxed{2}") == "2"


def test_extract_boxed_unbalanced_is_unparseable():
    # a continuation cut off mid-answer must NOT yield a garbage parse
    assert extract_boxed(r"ends truncated \boxed{\frac{1}{2") is None
    assert extract_boxed(r"\boxed{") is None


def test_extract_explicit_answer_has_no_number_fallback():
    """Outcome classification uses this: a truncated continuation full of
    stray numbers must parse as None (-> DISRUPTED), never as its last
    number (-> spurious REVERT)."""
    assert extract_explicit_answer("so 7 plus 3 gives 10, recompute 7...") is None
    assert extract_explicit_answer(r"thus \boxed{15}") == "15"
    assert extract_explicit_answer("Final answer: 12\n") == "12"
    assert extract_explicit_answer("") is None


def test_extract_answer_fallbacks():
    assert extract_answer(r"The final answer is \boxed{7}.") == "7"
    assert extract_answer("Final answer: 12\n") == "12"
    assert extract_answer("so we get 3 then 15 total") == "15"
    assert extract_answer("no numbers here") is None


def test_probe_answer_parsing():
    assert extract_probe_answer("42} whatever") == "42"
    assert extract_probe_answer(r"\frac{1}{2}") == r"\frac{1}{2}"  # no closing brace seen
    assert extract_probe_answer("   ") is None


def test_normalization():
    assert normalize_answer(r"\dfrac{1}{2}") == r"\frac{1}{2}"
    assert normalize_answer("$ 1,234,567 $") == "1234567"
    assert normalize_answer("45°") == "45"
    assert normalize_answer(r"\text{east}") == "east"


def test_normalization_drops_formatting_space_only():
    """A space touching punctuation/operators/backslashes is formatting and is
    removed; a space between two alphanumerics is content and is kept."""
    assert normalize_answer(r"\left( 3, 4 \right)") == "(3,4)"
    assert normalize_answer("( 3 , 1 )") == "(3,1)"
    assert normalize_answer("6 + 9i") == "6+9i"
    assert normalize_answer(r"70 \sqrt{2}") == r"70\sqrt{2}"
    assert normalize_answer("no solution") == "no solution"
    assert normalize_answer("1 2") == "1 2"


def test_answers_equal_latex_spacing():
    """Regression: LaTeX spacing differences must not read as wrong answers.
    Scoring these unequal systematically mislabels ~5% of MATH-500 chains and,
    because the failed-chain population is selected on 'parses AND wrong',
    admits every such false negative into it."""
    assert answers_equal(r"\left(3, \dfrac{\pi}{2}\right)", r"\left( 3, \frac{\pi}{2} \right)")
    assert answers_equal("(6, 31, -1)", "(6,31,-1)")
    assert answers_equal("(a + 5)(b + 2)", "(a+5)(b+2)")
    assert answers_equal(r"3\sqrt{5}", r"3 \sqrt{5}")
    # still conservative: distinct word answers do not merge
    assert not answers_equal("no solution", "nosolution")
    assert not answers_equal("12", "1 2")


def test_answers_equal_numeric():
    assert answers_equal("0.5", r"\frac{1}{2}")
    assert answers_equal("1/2", "0.5")
    assert answers_equal("1,000", "1000")
    assert not answers_equal("0.5", "0.6")
    assert not answers_equal(None, "1")


def test_answers_equal_strings():
    assert answers_equal("East", "east")
    assert answers_equal(r"\frac{\pi}{2}", r"\dfrac{\pi}{2}")
    assert not answers_equal(r"\frac{\pi}{2}", r"\frac{\pi}{3}")


def test_verifier_rule_based_and_cache(tmp_path):
    v = AnswerVerifier(tmp_path / "cache.jsonl")
    res = v.verify("p1", r"thus \boxed{\frac{2}{4}}", "0.5")
    assert res["correct"] and res["parsed"] == r"\frac{2}{4}"
    # cache hit returns the stored record without recompute
    v2 = AnswerVerifier(tmp_path / "cache.jsonl")
    assert v2.cache.get(problem_id="p1", answer_text=r"thus \boxed{\frac{2}{4}}") is not None
    res2 = v2.verify("p1", r"thus \boxed{\frac{2}{4}}", "0.5")
    assert res2["correct"] is True


def test_verifier_wrong_answer(tmp_path):
    v = AnswerVerifier(tmp_path / "cache.jsonl")
    assert v.verify("p2", r"\boxed{3}", "4")["correct"] is False
    assert v.verify("p3", "no parseable answer at all", "4")["correct"] is False
