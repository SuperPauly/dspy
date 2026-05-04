"""Tests for TextualFrequencyOptimizer.

All tests are deterministic and do not make live LM calls.
Paraphrase generation is handled by a fake paraphraser injected via dependency
injection, and word-frequency scoring is handled via the ``scorer`` override.
"""

import math
import sys
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

import dspy
from dspy.teleprompt import TextualFrequencyOptimizer
from dspy.teleprompt.textual_frequency_optimizer import (
    _deduplicate_preserve_order,
    _extract_paraphrases,
    _sentence_frequency_score,
    _tokenize_words,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_paraphraser(paraphrases: list[str]) -> Callable[..., Any]:
    """Return a callable that always produces the given paraphrases."""

    class FakeResult:
        pass

    def _paraphraser(instruction: str, count: int) -> Any:
        result = FakeResult()
        result.paraphrases = paraphrases
        return result

    return _paraphraser


def _constant_scorer(value: float) -> Callable[[str], float]:
    """Return a scorer that always returns *value* regardless of input."""
    return lambda _text: value


def _index_scorer(texts: list[str]) -> Callable[[str], float]:
    """Return a scorer that gives score = position of *text* in *texts* + 1."""

    def _score(text: str) -> float:
        try:
            return float(texts.index(text) + 1)
        except ValueError:
            return 0.0

    return _score


# ---------------------------------------------------------------------------
# Simple DSPy programs for testing
# ---------------------------------------------------------------------------


class SimplePredict(dspy.Module):
    """A module with a single Predict predictor."""

    def __init__(self, instructions: str = "Answer the question."):
        super().__init__()
        sig = dspy.Signature("question -> answer")
        sig = sig.with_instructions(instructions)
        self.predictor = dspy.Predict(sig)

    def forward(self, question: str) -> dspy.Prediction:
        return self.predictor(question=question)


class NestedModule(dspy.Module):
    """A module with two nested Predict predictors."""

    def __init__(self):
        super().__init__()
        sig_a = dspy.Signature("text -> summary").with_instructions("Summarize the text.")
        sig_b = dspy.Signature("summary -> sentiment").with_instructions("Classify the sentiment.")
        self.summarizer = dspy.Predict(sig_a)
        self.classifier = dspy.Predict(sig_b)

    def forward(self, text: str) -> dspy.Prediction:
        summary = self.summarizer(text=text)
        return self.classifier(summary=summary.summary)


# ---------------------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_from_teleprompt(self):
        from dspy.teleprompt import TextualFrequencyOptimizer as TFOptimizer

        assert TFOptimizer is TextualFrequencyOptimizer

    def test_from_dspy(self):
        assert hasattr(dspy, "TextualFrequencyOptimizer")
        assert dspy.TextualFrequencyOptimizer is TextualFrequencyOptimizer


# ---------------------------------------------------------------------------
# 2. Tokenisation tests
# ---------------------------------------------------------------------------


class TestTokenizeWords:
    def test_basic(self):
        assert _tokenize_words("Hello, world!") == ["hello", "world"]

    def test_punctuation_ignored(self):
        tokens = _tokenize_words("It's a test: with punctuation.")
        assert "." not in tokens
        assert ":" not in tokens
        assert "," not in tokens

    def test_case_normalised(self):
        tokens = _tokenize_words("CamelCase UPPER lower")
        assert tokens == ["camelcase", "upper", "lower"]

    def test_empty_string(self):
        assert _tokenize_words("") == []

    def test_only_punctuation(self):
        assert _tokenize_words("!!! ???") == []

    def test_apostrophe_contraction(self):
        tokens = _tokenize_words("it's fine")
        assert "it's" in tokens or ("it" in tokens and "s" in tokens)


# ---------------------------------------------------------------------------
# 3. Sentence frequency score tests
# ---------------------------------------------------------------------------


class TestSentenceFrequencyScore:
    def _fake_wf(self, word: str, lang: str) -> float:
        """Fake word_frequency returning fixed values."""
        freq_map = {"the": 0.1, "cat": 0.01, "sat": 0.001}
        return freq_map.get(word, 1e-6)

    def test_geometric_mean(self):
        # "the cat sat" -> freqs [0.1, 0.01, 0.001]
        expected = math.exp((math.log(0.1) + math.log(0.01) + math.log(0.001)) / 3)
        score = _sentence_frequency_score(
            "the cat sat",
            lang="en",
            min_word_frequency=1e-12,
            word_frequency_fn=self._fake_wf,
        )
        assert math.isclose(score, expected, rel_tol=1e-9)

    def test_empty_string_returns_zero(self):
        score = _sentence_frequency_score(
            "!!!",
            lang="en",
            min_word_frequency=1e-12,
            word_frequency_fn=self._fake_wf,
        )
        assert score == 0.0

    def test_min_word_frequency_floor(self):
        # unknown word defaults to 1e-6 via fake_wf; verify floor is respected
        floor = 1e-3
        score = _sentence_frequency_score(
            "unknownwordxyz",
            lang="en",
            min_word_frequency=floor,
            word_frequency_fn=self._fake_wf,
        )
        # 1e-6 < 1e-3, so floor kicks in -> score = exp(log(1e-3) / 1) = 1e-3
        assert math.isclose(score, floor, rel_tol=1e-9)

    def test_missing_wordfreq_raises_import_error(self):
        """Without custom scorer, missing wordfreq should raise ImportError."""
        # Temporarily remove wordfreq from sys.modules to simulate absence.
        original = sys.modules.get("wordfreq", None)
        sys.modules["wordfreq"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="wordfreq"):
                _sentence_frequency_score("hello world", lang="en", min_word_frequency=1e-12)
        finally:
            if original is None:
                sys.modules.pop("wordfreq", None)
            else:
                sys.modules["wordfreq"] = original


# ---------------------------------------------------------------------------
# 4. Paraphrase extraction tests
# ---------------------------------------------------------------------------


class TestExtractParaphrases:
    def _make_pred(self, value: Any) -> Any:
        result = MagicMock()
        result.paraphrases = value
        return result

    def test_list(self):
        pred = self._make_pred(["a", "b", "c"])
        assert _extract_paraphrases(pred) == ["a", "b", "c"]

    def test_tuple(self):
        pred = self._make_pred(("a", "b"))
        assert _extract_paraphrases(pred) == ["a", "b"]

    def test_newline_string(self):
        pred = self._make_pred("line one\nline two\nline three")
        assert _extract_paraphrases(pred) == ["line one", "line two", "line three"]

    def test_json_string_list(self):
        pred = self._make_pred('["alpha", "beta", "gamma"]')
        assert _extract_paraphrases(pred) == ["alpha", "beta", "gamma"]

    def test_strips_whitespace(self):
        pred = self._make_pred(["  hello  ", " world "])
        assert _extract_paraphrases(pred) == ["hello", "world"]

    def test_drops_empty_strings(self):
        pred = self._make_pred(["", "keep", "   ", "also keep"])
        assert _extract_paraphrases(pred) == ["keep", "also keep"]

    def test_deduplication(self):
        pred = self._make_pred(["same", "other", "same"])
        assert _extract_paraphrases(pred) == ["same", "other"]


# ---------------------------------------------------------------------------
# 5. Deduplicate helper
# ---------------------------------------------------------------------------


class TestDeduplicatePreserveOrder:
    def test_order_preserved(self):
        assert _deduplicate_preserve_order(["b", "a", "b", "c"]) == ["b", "a", "c"]

    def test_no_duplicates(self):
        assert _deduplicate_preserve_order(["x", "y", "z"]) == ["x", "y", "z"]

    def test_empty(self):
        assert _deduplicate_preserve_order([]) == []


# ---------------------------------------------------------------------------
# 6. Candidate selection tests
# ---------------------------------------------------------------------------


class TestCandidateSelection:
    """Verify that compile() selects the highest-scoring paraphrase."""

    def test_selects_best_by_score(self):
        paraphrases = ["low freq", "medium freq paraphrase", "high frequency common words"]
        # scorer returns position+1 so last element has highest score
        scorer = _index_scorer(paraphrases)
        fake_paraphraser = _make_fake_paraphraser(paraphrases)

        optimizer = TextualFrequencyOptimizer(
            num_candidates=3,
            paraphraser=fake_paraphraser,
            scorer=scorer,
        )
        program = SimplePredict("Original instruction.")
        compiled = optimizer.compile(program)

        selected = compiled.predictor.signature.instructions
        assert selected == "high frequency common words"

    def test_ties_keep_first(self):
        paraphrases = ["alpha", "beta", "gamma"]
        scorer = _constant_scorer(1.0)
        fake_paraphraser = _make_fake_paraphraser(paraphrases)

        optimizer = TextualFrequencyOptimizer(
            num_candidates=3,
            paraphraser=fake_paraphraser,
            scorer=scorer,
        )
        program = SimplePredict("Tie instruction.")
        compiled = optimizer.compile(program)

        # All scores equal; first candidate wins
        assert compiled.predictor.signature.instructions == "alpha"


# ---------------------------------------------------------------------------
# 7. compile() does not mutate the original
# ---------------------------------------------------------------------------


class TestNoMutation:
    def test_original_unchanged(self):
        original_instruction = "Do not change me."
        program = SimplePredict(original_instruction)
        paraphrases = ["Changed instruction one.", "Changed instruction two."]
        fake_paraphraser = _make_fake_paraphraser(paraphrases)
        scorer = _index_scorer(paraphrases)

        optimizer = TextualFrequencyOptimizer(
            num_candidates=2,
            paraphraser=fake_paraphraser,
            scorer=scorer,
        )
        compiled = optimizer.compile(program)

        assert program.predictor.signature.instructions == original_instruction
        assert compiled.predictor.signature.instructions != original_instruction

    def test_compiled_is_different_object(self):
        program = SimplePredict("Some instruction.")
        fake_paraphraser = _make_fake_paraphraser(["New instruction."])
        optimizer = TextualFrequencyOptimizer(
            num_candidates=1,
            paraphraser=fake_paraphraser,
            scorer=_constant_scorer(1.0),
        )
        compiled = optimizer.compile(program)
        assert compiled is not program


# ---------------------------------------------------------------------------
# 8. Fields preserved (input/output unchanged)
# ---------------------------------------------------------------------------


class TestFieldsPreserved:
    def test_input_output_fields_unchanged(self):
        program = SimplePredict("Original instruction.")
        original_input_fields = set(program.predictor.signature.input_fields.keys())
        original_output_fields = set(program.predictor.signature.output_fields.keys())

        fake_paraphraser = _make_fake_paraphraser(["A new instruction."])
        optimizer = TextualFrequencyOptimizer(
            num_candidates=1,
            paraphraser=fake_paraphraser,
            scorer=_constant_scorer(1.0),
        )
        compiled = optimizer.compile(program)

        compiled_input_fields = set(compiled.predictor.signature.input_fields.keys())
        compiled_output_fields = set(compiled.predictor.signature.output_fields.keys())

        assert compiled_input_fields == original_input_fields
        assert compiled_output_fields == original_output_fields
        # Only instructions changed
        assert compiled.predictor.signature.instructions == "A new instruction."


# ---------------------------------------------------------------------------
# 9. Nested module support
# ---------------------------------------------------------------------------


class TestNestedModules:
    def test_all_predictors_optimised(self):
        program = NestedModule()
        original_summary_instr = program.summarizer.signature.instructions
        original_classifier_instr = program.classifier.signature.instructions

        paraphrases_a = ["Summarise the provided text."]
        paraphrases_b = ["Determine the sentiment of the summary."]

        call_count = 0

        def _seq_paraphraser(instruction: str, count: int) -> Any:
            nonlocal call_count

            class FakeResult:
                pass

            r = FakeResult()
            if call_count == 0:
                r.paraphrases = paraphrases_a
            else:
                r.paraphrases = paraphrases_b
            call_count += 1
            return r

        optimizer = TextualFrequencyOptimizer(
            num_candidates=1,
            paraphraser=_seq_paraphraser,
            scorer=_constant_scorer(1.0),
        )
        compiled = optimizer.compile(program)

        # Original unchanged
        assert program.summarizer.signature.instructions == original_summary_instr
        assert program.classifier.signature.instructions == original_classifier_instr

        # Compiled updated
        assert compiled.summarizer.signature.instructions == "Summarise the provided text."
        assert compiled.classifier.signature.instructions == "Determine the sentiment of the summary."

    def test_metadata_attached(self):
        program = NestedModule()
        fake_paraphraser = _make_fake_paraphraser(["A paraphrase."])
        optimizer = TextualFrequencyOptimizer(
            num_candidates=1,
            paraphraser=fake_paraphraser,
            scorer=_constant_scorer(1.0),
        )
        compiled = optimizer.compile(program)

        assert hasattr(compiled, "textual_frequency_results")
        assert len(compiled.textual_frequency_results) == 2
        for entry in compiled.textual_frequency_results:
            assert "predictor_name" in entry
            assert "original_instruction" in entry
            assert "selected_instruction" in entry
            assert "candidate_scores" in entry


# ---------------------------------------------------------------------------
# 10. Compiled flag
# ---------------------------------------------------------------------------


class TestCompiledFlag:
    def test_compiled_flag_set(self):
        program = SimplePredict("Instruction.")
        fake_paraphraser = _make_fake_paraphraser(["New instruction."])
        optimizer = TextualFrequencyOptimizer(
            num_candidates=1,
            paraphraser=fake_paraphraser,
            scorer=_constant_scorer(1.0),
        )
        compiled = optimizer.compile(program)
        assert compiled._compiled is True


# ---------------------------------------------------------------------------
# 11. include_original behaviour
# ---------------------------------------------------------------------------


class TestIncludeOriginal:
    def test_original_in_candidates_when_true(self):
        original_instr = "Original."
        paraphrases = ["Paraphrase A."]
        # With include_original=True, original goes first; scorer assigns 1 to it
        scorer = _index_scorer([original_instr, "Paraphrase A."])
        fake_paraphraser = _make_fake_paraphraser(paraphrases)

        optimizer = TextualFrequencyOptimizer(
            num_candidates=1,
            include_original=True,
            paraphraser=fake_paraphraser,
            scorer=scorer,
        )
        program = SimplePredict(original_instr)
        compiled = optimizer.compile(program)
        # original_instr scores 1, paraphrase scores 2 -> paraphrase wins (higher)
        assert compiled.predictor.signature.instructions == "Paraphrase A."


# ---------------------------------------------------------------------------
# 12. Error tests
# ---------------------------------------------------------------------------


class TestErrors:
    def test_invalid_num_candidates(self):
        with pytest.raises(ValueError, match="num_candidates"):
            TextualFrequencyOptimizer(num_candidates=0)

    def test_empty_lang(self):
        with pytest.raises(ValueError, match="lang"):
            TextualFrequencyOptimizer(lang="")

    def test_invalid_min_word_frequency(self):
        with pytest.raises(ValueError, match="min_word_frequency"):
            TextualFrequencyOptimizer(min_word_frequency=0.0)

    def test_no_predictors_raises(self):
        class EmptyModule(dspy.Module):
            def forward(self):
                pass

        optimizer = TextualFrequencyOptimizer(
            num_candidates=1,
            paraphraser=_make_fake_paraphraser(["x"]),
            scorer=_constant_scorer(1.0),
        )
        with pytest.raises(ValueError, match=r"no dspy\.Predict predictors"):
            optimizer.compile(EmptyModule())

    def test_no_valid_paraphrases_raises(self):
        # Paraphraser returns only empty/whitespace strings
        fake_paraphraser = _make_fake_paraphraser(["", "   ", "  "])
        optimizer = TextualFrequencyOptimizer(
            num_candidates=3,
            paraphraser=fake_paraphraser,
            scorer=_constant_scorer(1.0),
        )
        with pytest.raises(ValueError, match="No valid paraphrases"):
            optimizer.compile(SimplePredict("Some instruction."))

    def test_missing_wordfreq_raises_import_error(self):
        """Without a custom scorer, absent wordfreq yields ImportError."""
        # Patch sys.modules to simulate absence of wordfreq
        original = sys.modules.get("wordfreq", None)
        sys.modules["wordfreq"] = None  # type: ignore[assignment]
        try:
            from dspy.teleprompt.textual_frequency_optimizer import _load_word_frequency

            with pytest.raises(ImportError, match="wordfreq"):
                _load_word_frequency()
        finally:
            if original is None:
                sys.modules.pop("wordfreq", None)
            else:
                sys.modules["wordfreq"] = original
