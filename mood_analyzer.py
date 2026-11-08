# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import (
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    NEGATIVE_SITUATIONS,
    OBLIGATION_PHRASES,
    SARCASTIC_OPENERS,
)


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    # Emojis, emoticons, and slang carry stronger signals than ordinary words,
    # so they get larger weights. Plain sentiment words count as +1 / -1.
    # Keys must match what preprocess() produces (lowercase, emoticons
    # canonicalized like ":-)))" -> ":)", apostrophes removed from words).
    STRONG_SIGNALS: Dict[str, int] = {
        # Text emoticons
        ":)": 2, ":d": 2, ";)": 2, ";d": 2, ":p": 2, "=)": 2, "<3": 3,
        ":(": -2, ":'(": -3, ":/": -2, ":|": -1, "</3": -3,
        # Positive emoji
        "😂": 2, "🤣": 2, "😊": 2, "😁": 2, "😄": 2, "😍": 3, "🥰": 3,
        "❤": 3, "🔥": 2, "🎉": 2, "👍": 2, "🙌": 2, "✨": 1,
        # "💀" usually means "I'm dying (laughing)" online, so it's positive.
        "💀": 2,
        # Negative emoji
        "😢": -3, "😭": -2, "😞": -2, "😔": -2, "😡": -3, "😠": -3,
        "🙄": -2, "😩": -2, "😤": -2, "💔": -3, "👎": -2,
        # Positive slang
        "lol": 1, "lmao": 2, "haha": 1, "yay": 2, "goat": 2, "fire": 2,
        "lit": 2, "slay": 2, "bussin": 2, "vibes": 1, "w": 2,
        # Negative slang
        "mid": -2, "ugh": -2, "smh": -2, "yikes": -2, "meh": -1, "cringe": -2,
        "trash": -2, "sucks": -2, "l": -2, "bruh": -1,
    }

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
        strong_signals: Optional[Dict[str, int]] = None,
        negative_situations: Optional[List[str]] = None,
        obligation_phrases: Optional[List[Tuple[str, ...]]] = None,
        sarcastic_openers: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS
        strong_signals = strong_signals if strong_signals is not None else self.STRONG_SIGNALS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)
        self.strong_signals = {k.lower(): v for k, v in strong_signals.items()}

        # Sarcasm cues: bad situations and "forced to" phrases.
        if negative_situations is None:
            negative_situations = NEGATIVE_SITUATIONS
        if obligation_phrases is None:
            obligation_phrases = OBLIGATION_PHRASES
        self.negative_situations = set(w.lower() for w in negative_situations)
        self.obligation_phrases = [
            tuple(w.lower() for w in phrase) for phrase in obligation_phrases
        ]
        if sarcastic_openers is None:
            sarcastic_openers = SARCASTIC_OPENERS
        self.sarcastic_openers = set(w.lower() for w in sarcastic_openers)

    def _is_known(self, token: str) -> bool:
        """True if the token carries any sentiment in our lexicons."""
        return (
            token in self.positive_words
            or token in self.negative_words
            or token in self.strong_signals
        )

    def token_weight(self, token: str) -> int:
        """
        Sentiment weight of a single token. Strong signals (emojis, emoticons,
        slang) use their own weight; ordinary words are +1 / -1.
        """
        if token in self.strong_signals:
            return self.strong_signals[token]
        if token in self.positive_words:
            return 1
        if token in self.negative_words:
            return -1
        return 0

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    # Text emoticons such as ":)", ":-(", ";D", ":'(", ":p", "<3", "</3".
    # The lookbehind stops matches inside words or URLs (e.g. "http://").
    _EMOTICON = r"(?<![a-z0-9])(?:[:;=]['\-^]?[()\[\]dpo/\\|*3]+|<\/?3+)"
    # Unicode emoji (pictographs, symbols, dingbats, skin-tone modifiers).
    _EMOJI = r"[\U0001F300-\U0001FAFF☀-➿]"
    # Words, allowing internal apostrophes ("don't", "i'm").
    _WORD = r"[a-z0-9]+(?:'[a-z0-9]+)*"
    _TOKEN_RE = re.compile(f"({_EMOTICON})|({_EMOJI})|({_WORD})")

    def _normalize_emoticon(self, emoticon: str) -> str:
        """Canonicalize emoticons: ":-)))" -> ":)", ":'(" -> ":'(", "<333" -> "<3"."""
        if emoticon.startswith("<"):
            return "</3" if "/" in emoticon else "<3"
        eyes, rest = emoticon[0], emoticon[1:]
        # Keep the tear ("'") since ":'(" means something different; drop noses.
        tear = "'" if rest.startswith("'") else ""
        rest = rest.lstrip("'-^")
        # Collapse repeated mouth characters (":)))" -> ":)").
        mouth = re.sub(r"(.)\1+", r"\1", rest)
        return eyes + tear + mouth

    def _normalize_repeats(self, word: str) -> str:
        """
        Shrink runs of 3+ identical characters down to 2 ("soooo" -> "soo").
        If that isn't a known sentiment word but collapsing runs to a single
        character is, use the single-character form ("happyyyy" -> "happy").
        """
        doubled = re.sub(r"(.)\1{2,}", r"\1\1", word)
        if self._is_known(doubled):
            return doubled
        # Only shrink the runs that were stretched (3+), so real double
        # letters like the "pp" in "happy" are left alone.
        single = re.sub(r"(.)\1{2,}", r"\1", word)
        if single != doubled and self._is_known(single):
            return single
        return doubled

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          - Lowercase and normalize curly apostrophes
          - Pull out emoticons (":)", ":-(") and emoji ("🥲", "😂") as their
            own tokens, so they survive punctuation removal
          - Remove all other punctuation (apostrophes inside words are
            dropped, so "don't" -> "dont")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower().replace("’", "'").replace("️", "")

        tokens: List[str] = []
        for emoticon, emoji, word in self._TOKEN_RE.findall(cleaned):
            if emoticon:
                tokens.append(self._normalize_emoticon(emoticon))
            elif emoji:
                tokens.append(emoji)
            elif word:
                tokens.append(self._normalize_repeats(word.replace("'", "")))

        return tokens

    # ---------------------------------------------------------------------
    # Sarcasm detection
    # ---------------------------------------------------------------------

    # How many tokens after a positive word to search for a sarcasm cue.
    SARCASM_WINDOW = 5

    def _is_sarcastic_context(self, tokens: List[str], i: int) -> bool:
        """
        True if the positive token at position i looks sarcastic: it comes
        right after a sarcastic opener ("oh great, more work"), or it is
        followed within SARCASM_WINDOW tokens by a bad situation
        ("love getting stuck in traffic") or an obligation phrase
        ("so happy I have to take orders").
        """
        if i > 0 and tokens[i - 1] in self.sarcastic_openers:
            return True
        after = tokens[i + 1 : i + 1 + self.SARCASM_WINDOW]
        if any(t in self.negative_situations for t in after):
            return True
        for start in range(len(after)):
            for phrase in self.obligation_phrases:
                if tuple(after[start : start + len(phrase)]) == phrase:
                    return True
        return False

    def contextual_weights(self, tokens: List[str]) -> List[int]:
        """
        Sentiment weight of each token, taking context into account.
        A positive token in a sarcastic context has its sign flipped.
        """
        weights: List[int] = []
        for i, token in enumerate(tokens):
            weight = self.token_weight(token)
            if weight > 0 and self._is_sarcastic_context(tokens, i):
                weight = -weight
            weights.append(weight)
        return weights

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score by 1; negative words decrease it by 1.

        Modeling improvement: emojis, emoticons, and slang (":)", "lol", "💀",
        "mid") are treated as strong signals with larger weights from
        STRONG_SIGNALS, so a single "😭" or "</3" outweighs one plain word.
        Every occurrence counts, so "😂😂😂" adds more than one "😂".

        Sarcasm: a positive word followed shortly by a bad situation or an
        obligation ("I love getting stuck in traffic", "so happy I have to
        work") counts as negative instead of positive.
        """
        return sum(self.contextual_weights(self.preprocess(text)))

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"
          - both positive and negative signals with |score| <= 1 -> "mixed"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score = self.score_text(text)

        # "mixed": the text has both positive and negative signals that
        # roughly cancel out (e.g. "tired but happy").
        weights = self.contextual_weights(self.preprocess(text))
        has_positive = any(w > 0 for w in weights)
        has_negative = any(w < 0 for w in weights)
        if has_positive and has_negative and abs(score) <= 1:
            return "mixed"

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)
        weights = self.contextual_weights(tokens)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        sarcasm_flips: List[str] = []

        for token, weight in zip(tokens, weights):
            if weight > 0:
                positive_hits.append(token)
            elif weight < 0:
                # A positive word whose sign was flipped by sarcasm detection.
                if self.token_weight(token) > 0:
                    sarcasm_flips.append(token)
                negative_hits.append(token)

        explanation = (
            f"Score = {sum(weights)} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
        if sarcasm_flips:
            explanation += f" [sarcasm: flipped {sarcasm_flips} to negative]"
        return explanation
