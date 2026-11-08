"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
]

# Situations almost nobody enjoys. A positive word followed closely by one of
# these ("I love getting stuck in traffic") is treated as sarcasm.
# Words must match preprocess() output (lowercase, no apostrophes).
NEGATIVE_SITUATIONS = [
    "traffic",
    "stuck",
    "homework",
    "monday",
    "mondays",
    "waiting",
    "overtime",
    "closing",
    "deadline",
    "deadlines",
    "delayed",
    "cancelled",
    "canceled",
    "chores",
    "detention",
    "dentist",
    "assighment",
]

# Phrases meaning "being forced to do something" ("so happy I have to work").
# Each phrase is a tuple of consecutive tokens.
OBLIGATION_PHRASES = [
    ("have", "to"),
    ("has", "to"),
    ("got", "to"),
    ("gotta",),
]

# Words that make the positive word right after them sarcastic
# ("oh great, more work", "oh wonderful", "oh joy").
SARCASTIC_OPENERS = [
    "oh",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "This shift is boring",
    "The Sun feels nice today",
    "I'm so happy I have to take orders 10 minutes befor closing",
    "This show is mid",
    "This is very annoying",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "negative",  # "This shift is boring"
    "positive",  # "The sun feels nice today"
    "negative",  # "I'm so happy I have to take orders 10 minutes befor closing"
    "negative",  # "This show is mid"
    "negative",  # This is very annoying"
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
