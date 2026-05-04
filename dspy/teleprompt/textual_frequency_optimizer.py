"""TextualFrequencyOptimizer: zero-shot instruction optimiser based on textual-frequency scoring.

This module implements a textual-frequency-based zero-shot instruction optimization method.
For each Predict predictor in a DSPy program, the optimizer generates a set of semantically
equivalent paraphrases of the existing instruction and selects the one with the highest
geometric-mean unigram frequency (using the ``wordfreq`` library).  No labelled data or
task metric is required.

Requires the optional dependency ``wordfreq``::

    pip install wordfreq
"""

import json
import logging
import math
from collections.abc import Callable, Iterable
from typing import Any

import regex

import dspy
from dspy.signatures.signature import Signature
from dspy.teleprompt.teleprompt import Teleprompter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal DSPy signature used for paraphrase generation
# ---------------------------------------------------------------------------


class GenerateInstructionParaphrases(dspy.Signature):
    """Generate semantically equivalent paraphrases of an instruction."""

    instruction: str = dspy.InputField(desc="The original instruction.")
    count: int = dspy.InputField(desc="The exact number of paraphrases to generate.")
    paraphrases: list[str] = dspy.OutputField(
        desc="A list of semantically equivalent paraphrases. Do not add new requirements or remove constraints.",
    )


# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------


def _load_word_frequency() -> Callable[..., float]:
    """Return ``wordfreq.word_frequency``, raising a helpful error if missing."""
    try:
        from wordfreq import word_frequency  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "TextualFrequencyOptimizer requires the optional dependency `wordfreq`. "
            "Install it with `pip install wordfreq`."
        ) from exc
    return word_frequency


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

# Matches Unicode letters (with optional apostrophe-contracted suffix) or Unicode digits.
_TOKEN_RE = regex.compile(r"\p{L}+(?:['']\p{L}+)?|\p{N}+")


def _tokenize_words(text: str) -> list[str]:
    """Return lower-cased word tokens from *text*, ignoring punctuation."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------


def _sentence_frequency_score(
    text: str,
    *,
    lang: str,
    min_word_frequency: float,
    word_frequency_fn: Callable[..., float] | None = None,
) -> float:
    """Return the geometric-mean unigram frequency of *text*.

    If *text* contains no tokens, returns 0.0.

    Args:
        text: The sentence to score.
        lang: BCP 47 language code passed to ``wordfreq``.
        min_word_frequency: Floor applied before taking ``log`` to avoid ``log(0)``.
        word_frequency_fn: Optional override for testing; defaults to
            ``wordfreq.word_frequency``.
    """
    if word_frequency_fn is None:
        word_frequency_fn = _load_word_frequency()
    words = _tokenize_words(text)
    if not words:
        return 0.0
    freqs = [max(word_frequency_fn(w, lang), min_word_frequency) for w in words]
    return math.exp(sum(math.log(f) for f in freqs) / len(freqs))


# ---------------------------------------------------------------------------
# Paraphrase extraction helpers
# ---------------------------------------------------------------------------


def _deduplicate_preserve_order(items: Iterable[str]) -> list[str]:
    """Return *items* with duplicates removed, preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _extract_paraphrases(prediction: Any) -> list[str]:
    """Robustly extract a list of paraphrase strings from a DSPy prediction.

    Handles:
    - ``list[str]``
    - ``tuple[str, ...]``
    - newline-delimited string
    - JSON-like string containing a list
    """
    raw = prediction.paraphrases

    if isinstance(raw, (list, tuple)):
        candidates = [str(item) for item in raw]
    elif isinstance(raw, str):
        # Try JSON parse first
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    candidates = [str(item) for item in parsed]
                else:
                    candidates = stripped.splitlines()
            except json.JSONDecodeError:
                candidates = stripped.splitlines()
        else:
            candidates = stripped.splitlines()
    else:
        candidates = [str(raw)]

    cleaned = [c.strip() for c in candidates]
    cleaned = [c for c in cleaned if c]
    return _deduplicate_preserve_order(cleaned)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------


class TextualFrequencyOptimizer(Teleprompter):
    """Zero-shot instruction optimizer based on textual-frequency scoring.

    This optimizer implements a textual-frequency-based zero-shot instruction
    optimization method.  For each ``dspy.Predict`` predictor in the compiled
    program the optimizer:

    1. Extracts the current signature instructions.
    2. Generates ``num_candidates`` semantically equivalent paraphrases using a
       DSPy ``Predict`` module (or a user-supplied callable).
    3. Scores every paraphrase with the geometric mean of word-level unigram
       frequencies provided by the ``wordfreq`` library.
    4. Replaces the predictor's instructions with the highest-scoring paraphrase.

    .. note::
        This optimizer relies on the configured language model to generate
        semantically equivalent paraphrases.  Semantic equivalence is
        instructed but not guaranteed.  Always measure task performance with a
        DSPy metric after optimizing.

    .. note::
        This optimizer requires the optional dependency ``wordfreq``::

            pip install wordfreq

    Args:
        num_candidates: Number of paraphrase candidates to generate per
            predictor.  Must be ``>= 1``.  Defaults to ``10``.
        lang: BCP 47 language code passed to ``wordfreq.word_frequency``.
            Defaults to ``"en"``.
        min_word_frequency: Floor applied to word frequencies before taking
            ``log``, avoiding ``log(0)``.  Must be ``> 0``.
            Defaults to ``1e-12``.
        include_original: When ``True``, the original instruction is added to
            the candidate pool before scoring.  Defaults to ``False``.
        paraphraser: Optional DSPy module (or callable) used to generate
            paraphrases.  When ``None``, a ``dspy.Predict`` module with
            ``GenerateInstructionParaphrases`` is used.  The callable must
            accept keyword arguments ``instruction`` and ``count`` and return
            an object with a ``paraphrases`` attribute.
        temperature: Temperature used by the default ``dspy.Predict``
            paraphraser.  Ignored when *paraphraser* is provided explicitly.
            Defaults to ``0.7``.
        scorer: Optional callable ``(text: str) -> float`` used to score
            paraphrase candidates.  Intended for testing; when ``None`` the
            ``wordfreq``-based scorer is used.
    """

    def __init__(
        self,
        num_candidates: int = 10,
        lang: str = "en",
        min_word_frequency: float = 1e-12,
        include_original: bool = False,
        paraphraser: dspy.Module | None = None,
        temperature: float = 0.7,
        scorer: Callable[[str], float] | None = None,
    ) -> None:
        if num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {num_candidates!r}")
        if not lang:
            raise ValueError("lang must be a non-empty string")
        if min_word_frequency <= 0:
            raise ValueError(f"min_word_frequency must be > 0, got {min_word_frequency!r}")

        self.num_candidates = num_candidates
        self.lang = lang
        self.min_word_frequency = min_word_frequency
        self.include_original = include_original
        self.temperature = temperature

        self.paraphraser: dspy.Module | Callable[..., Any] = paraphraser or dspy.Predict(
            GenerateInstructionParaphrases,
            n=1,
            temperature=temperature,
        )

        # Private scorer override (primarily for testing).
        self._scorer: Callable[[str], float] | None = scorer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score(self, text: str) -> float:
        if self._scorer is not None:
            return self._scorer(text)
        return _sentence_frequency_score(
            text,
            lang=self.lang,
            min_word_frequency=self.min_word_frequency,
        )

    def _optimize_predictor_signature(
        self,
        name: str,
        predictor: "dspy.Predict",
    ) -> dict[str, Any]:
        """Optimize the signature instructions of a single predictor.

        Returns a metadata dict describing the optimization result.
        """
        original_instruction: str = predictor.signature.instructions

        prediction = self.paraphraser(instruction=original_instruction, count=self.num_candidates)
        paraphrases = _extract_paraphrases(prediction)

        if self.include_original:
            paraphrases = _deduplicate_preserve_order([original_instruction, *paraphrases])

        if not paraphrases:
            raise ValueError(
                f"No valid paraphrases were generated for predictor '{name}'. "
                "Ensure the configured language model is responding correctly."
            )

        scored: list[tuple[str, float]] = [(p, self._score(p)) for p in paraphrases]
        best_text, best_score = max(scored, key=lambda x: x[1])

        updated_sig: Signature = predictor.signature.with_instructions(best_text)
        predictor.signature = updated_sig

        logger.debug(
            "TextualFrequencyOptimizer: predictor '%s' -> selected instruction (score=%.6f): %r",
            name,
            best_score,
            best_text,
        )

        return {
            "predictor_name": name,
            "original_instruction": original_instruction,
            "selected_instruction": best_text,
            "candidate_scores": scored,
        }

    # ------------------------------------------------------------------
    # Public compile() method
    # ------------------------------------------------------------------

    def compile(
        self,
        student: dspy.Module,
        *,
        trainset: list[dspy.Example] | None = None,
        teacher: dspy.Module | None = None,
        valset: list[dspy.Example] | None = None,
        **kwargs: Any,
    ) -> dspy.Module:
        """Compile *student* by rewriting predictor instructions.

        This method deep-copies *student*, rewrites each predictor's signature
        instructions with the highest-frequency paraphrase, and returns the
        compiled copy.  The original *student* is never mutated.

        Args:
            student: The DSPy program to optimize.
            trainset: Unused.  Kept for API compatibility.
            teacher: Unused.  Kept for API compatibility.
            valset: Unused.  Kept for API compatibility.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A compiled copy of *student* with updated signature instructions.

        Raises:
            ValueError: If *student* contains no ``dspy.Predict`` predictors.
            ValueError: If paraphrase generation yields no valid candidates.
            ImportError: If ``wordfreq`` is not installed (and no custom
                *scorer* was supplied).
        """
        compiled = student.deepcopy()

        named_preds = compiled.named_predictors()
        if not named_preds:
            raise ValueError(
                "TextualFrequencyOptimizer: the student program contains no dspy.Predict predictors. "
                "Nothing to optimize."
            )

        results: list[dict[str, Any]] = []
        for name, predictor in named_preds:
            result = self._optimize_predictor_signature(name, predictor)
            results.append(result)

        compiled._compiled = True
        compiled.textual_frequency_results = results

        return compiled
