"""
Sentence builder for ISL Sign-to-Speech.

Collects stable recognized words into an ordered sequence and returns
the accumulated sequence when the user indicates the sentence is
complete. Held-sign suppression is handled by the recognition layer.

Labels are normalized to UPPERCASE internally so that "Beautiful",
"beautiful", and "BEAUTIFUL" are all treated as the same word.
"""

from __future__ import annotations


class SentenceBuilder:
    """Accumulates recognized sign words into a sentence."""

    def __init__(self) -> None:
        self._words: list[str] = []

    # ── Public API ──────────────────────────────────────────────

    def add_word(self, word: str) -> bool:
        """Add a recognized word while preserving repetitions.

        Parameters
        ----------
        word
            The recognized sign label (any casing).

        Returns
        -------
        ``True`` if the normalized word was added. ``False`` only when
        the input is empty.
        """
        normalized = word.strip().upper()

        if not normalized:
            return False

        self._words.append(normalized)
        return True

    def clear(self) -> None:
        """Remove all accumulated words."""
        self._words.clear()

    def complete(self) -> list[str]:
        """Return the current word list and reset for the next sentence.

        Returns
        -------
        A copy of the accumulated words.  The internal buffer is cleared
        after this call.
        """
        words = list(self._words)
        self._words.clear()
        return words

    def get_words(self) -> list[str]:
        """Return a copy of the current word list without clearing."""
        return list(self._words)

    def get_display_string(self) -> str:
        """Return a human-readable arrow-separated string.

        Example::

            "HELLO → BEAUTIFUL → SAD"

        Returns ``"(empty)"`` when no words have been collected.
        """
        if not self._words:
            return "(empty)"
        return " -> ".join(self._words)

    def is_empty(self) -> bool:
        """Return ``True`` if no words have been collected."""
        return len(self._words) == 0

    def __len__(self) -> int:
        return len(self._words)

    def __repr__(self) -> str:
        return f"SentenceBuilder(words={self._words!r})"
