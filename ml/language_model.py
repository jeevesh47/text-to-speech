"""
Language model integration for ISL Sign-to-Speech.

Converts a sequence of recognized ISL glosses into a natural English
sentence.  The LSTM / sentence builder remain responsible for the
gloss sequence.  This module only runs when a sentence is finished.

Providers (selected via ``LLM_PROVIDER``):

    google    — Google Gemini  (``GEMINI_API_KEY``)
    openai    — OpenAI         (``OPENAI_API_KEY``)
    groq      — Groq           (``GROQ_API_KEY``)
    fallback  — local grammar rules, no API

If ``LLM_PROVIDER`` is unset, Gemini is used when ``GEMINI_API_KEY``
is set, otherwise OpenAI when ``OPENAI_API_KEY`` is set, otherwise
Groq when ``GROQ_API_KEY`` is set, otherwise the local fallback.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Sequence

from dotenv import load_dotenv


load_dotenv()


# ── Prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an English sentence generation component for an Indian "
    "Sign Language translation system. The input is a sequence of "
    "recognized ISL glosses, NOT normal English. Convert the gloss "
    "sequence into natural, grammatically correct English. Preserve "
    "the intended meaning. Add necessary English grammar such as "
    "articles, auxiliary verbs, pronouns, prepositions, tense, and "
    "natural word order when appropriate. Do not invent information "
    "that is not supported by the input. Do not add extra people, "
    "places, reasons, or events. Return only the final English sentence."
)

_FEW_SHOT_EXAMPLES = (
    "ISL: I GO SCHOOL TOMORROW\n"
    "English: I am going to school tomorrow.\n"
    "\n"
    "ISL: I HELP FRIEND TODAY\n"
    "English: I am helping my friend today.\n"
    "\n"
    "ISL: I HAPPY\n"
    "English: I am happy.\n"
    "\n"
    "ISL: I SAD\n"
    "English: I am sad.\n"
    "\n"
    "ISL: I GO HOME\n"
    "English: I am going home.\n"
    "\n"
    "ISL: FRIEND HELP ME\n"
    "English: My friend is helping me.\n"
    "\n"
    "ISL: I WILL GO SCHOOL TOMORROW\n"
    "English: I will go to school tomorrow.\n"
    "\n"
    "ISL: HELLO\n"
    "English: Hello.\n"
    "\n"
    "ISL: GO HOME\n"
    "English: Go home.\n"
    "\n"
    "ISL: THANK YOU\n"
    "English: Thank you."
)


# ── Local grammar tables (extensible) ────────────────────────

_GREETINGS = {
    ("HELLO",): "Hello.",
    ("HI",): "Hi.",
    ("BYE",): "Goodbye.",
    ("GOODBYE",): "Goodbye.",
    ("THANKS",): "Thanks.",
    ("THANKYOU",): "Thank you.",
    ("THANK", "YOU"): "Thank you.",
    ("PLEASE",): "Please.",
    ("YES",): "Yes.",
    ("NO",): "No.",
    ("OK",): "Okay.",
    ("OKAY",): "Okay.",
}

_TIME_WORDS = {
    "TODAY": "today",
    "TOMORROW": "tomorrow",
    "YESTERDAY": "yesterday",
    "NOW": "now",
}

_STATE_WORDS = {
    "HAPPY",
    "SAD",
    "ANGRY",
    "TIRED",
    "SICK",
    "HUNGRY",
    "THIRSTY",
    "FINE",
    "GOOD",
    "BAD",
    "BEAUTIFUL",
    "UGLY",
    "LOUD",
    "QUIET",
    "DEAF",
    "BLIND",
}

_FUTURE_MARKERS = {"WILL"}

_SUBJECTS = {
    "I": "I",
    "YOU": "You",
    "WE": "We",
    "THEY": "They",
    "HE": "He",
    "SHE": "She",
}

_OBJECT_PRONOUNS = {
    "ME": "me",
    "YOU": "you",
    "HIM": "him",
    "HER": "her",
    "US": "us",
    "THEM": "them",
}

# Family / people nouns often take a possessive when the subject is "I".
_PEOPLE_NOUNS = {
    "FRIEND": "friend",
    "MOTHER": "mother",
    "FATHER": "father",
    "MOM": "mom",
    "DAD": "dad",
    "BROTHER": "brother",
    "SISTER": "sister",
    "TEACHER": "teacher",
    "DOCTOR": "doctor",
}

# Verb lemma -> (present participle, base form)
_VERBS = {
    "GO": ("going", "go"),
    "COME": ("coming", "come"),
    "HELP": ("helping", "help"),
    "WANT": ("wanting", "want"),
    "NEED": ("needing", "need"),
    "LIKE": ("liking", "like"),
    "LOVE": ("loving", "love"),
    "SEE": ("seeing", "see"),
    "EAT": ("eating", "eat"),
    "DRINK": ("drinking", "drink"),
    "LEARN": ("learning", "learn"),
    "STUDY": ("studying", "study"),
    "WORK": ("working", "work"),
    "PLAY": ("playing", "play"),
    "READ": ("reading", "read"),
    "WRITE": ("writing", "write"),
    "SLEEP": ("sleeping", "sleep"),
    "WAIT": ("waiting", "wait"),
    "GIVE": ("giving", "give"),
    "TAKE": ("taking", "take"),
    "MAKE": ("making", "make"),
    "BUY": ("buying", "buy"),
    "CALL": ("calling", "call"),
    "ASK": ("asking", "ask"),
    "TELL": ("telling", "tell"),
    "KNOW": ("knowing", "know"),
    "THINK": ("thinking", "think"),
    "LIVE": ("living", "live"),
}

_STATIVE_VERBS = {"WANT", "NEED", "LIKE", "LOVE", "KNOW"}

# Motion verbs take "to" before a destination, except a few nouns.
_MOTION_VERBS = {"GO", "COME"}
_NO_PREPOSITION = {"HOME", "HERE", "THERE", "INSIDE", "OUTSIDE", "UP", "DOWN"}


# ── Public API ───────────────────────────────────────────────

def get_active_provider() -> str:
    """Return the active provider: google, openai, groq, or fallback."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"google", "gemini"}:
        return "google"
    if explicit in {"openai"}:
        return "openai"
    if explicit in {"groq"}:
        return "groq"
    if explicit in {"fallback", "none"}:
        return "fallback"

    if os.getenv("GEMINI_API_KEY"):
        return "google"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GROQ_API_KEY"):
        return "groq"

    return "fallback"


def generate_sentence(words: list[str]) -> str:
    """Convert recognized ISL glosses into a natural English sentence.

    Parameters
    ----------
    words
        Ordered glosses from the sentence builder, e.g.
        ``["I", "GO", "SCHOOL", "TOMORROW"]``.

    Returns
    -------
    A single English sentence.  If the LLM is unavailable, a local
    grammar fallback is used.  On any LLM error the fallback is used.
    """
    cleaned = _clean_words(words)
    if not cleaned:
        return ""

    fallback = _fallback_sentence(cleaned)
    provider = get_active_provider()

    if provider == "google":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if key:
            try:
                result = _generate_with_gemini(cleaned, key)
                if result:
                    return result
            except Exception as exc:
                print(
                    f"[LanguageModel] Gemini API error ({exc}). "
                    "Using fallback."
                )

    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if key:
            try:
                result = _generate_with_openai(cleaned, key)
                if result:
                    return result
            except Exception as exc:
                print(
                    f"[LanguageModel] OpenAI API error ({exc}). "
                    "Using fallback."
                )

    elif provider == "groq":
        key = os.getenv("GROQ_API_KEY", "").strip()
        if key:
            try:
                result = _generate_with_groq(cleaned, key)
                if result:
                    return result
            except Exception as exc:
                print(
                    f"[LanguageModel] Groq API error ({exc}). "
                    "Using fallback."
                )

    return fallback


# ── Local fallback ───────────────────────────────────────────

def _clean_words(words: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for word in words:
        token = str(word).strip().upper()
        if token:
            cleaned.append(token)
    return cleaned


def _fallback_sentence(words: Sequence[str]) -> str:
    """Deterministic ISL-gloss to English conversion."""
    tokens = list(words)
    if not tokens:
        return ""

    greeting = _GREETINGS.get(tuple(tokens))
    if greeting:
        return greeting

    time_phrase = ""
    while tokens and tokens[-1] in _TIME_WORDS:
        time_word = _TIME_WORDS[tokens.pop()]
        if time_phrase:
            time_phrase = f"{time_word} {time_phrase}"
        else:
            time_phrase = time_word

    if not tokens:
        return _finish(time_phrase.capitalize() if time_phrase else "")

    future = False
    subject_key: str | None = None
    subject_text: str | None = None

    if tokens[0] in _SUBJECTS:
        subject_key = tokens.pop(0)
        subject_text = _SUBJECTS[subject_key]
    elif tokens[0] == "ME":
        subject_key = "I"
        subject_text = "I"
        tokens.pop(0)

    if tokens and tokens[0] in _FUTURE_MARKERS:
        tokens.pop(0)
        future = True

    if not tokens:
        return _finish(_join_parts(subject_text, time_phrase))

    # I HAPPY / I SAD
    if all(token in _STATE_WORDS for token in tokens):
        states = " and ".join(token.lower() for token in tokens)
        if subject_text:
            be = _be_form(subject_key or "I")
            return _finish(_join_parts(f"{subject_text} {be} {states}", time_phrase))
        return _finish(_join_parts(states.capitalize(), time_phrase))

    # Pronoun/noun + verb + objects
    if tokens[0] in _VERBS:
        verb = tokens.pop(0)
        objects = tokens
        return _verb_sentence(
            subject_key,
            subject_text,
            verb,
            objects,
            future,
            time_phrase,
        )

    # FRIEND HELP ME  (noun subject)
    if len(tokens) >= 2 and tokens[1] in _VERBS:
        noun = tokens.pop(0)
        if tokens and tokens[0] in _FUTURE_MARKERS:
            tokens.pop(0)
            future = True
        verb = tokens.pop(0)
        objects = tokens
        subject_text = _noun_as_subject(noun)
        subject_key = "HE"
        return _verb_sentence(
            subject_key,
            subject_text,
            verb,
            objects,
            future,
            time_phrase,
        )

    # Unknown pattern: readable join, not raw ALL-CAPS glosses.
    pretty = [tokens[0].capitalize(), *[t.lower() for t in tokens[1:]]]
    if subject_text:
        pretty = [subject_text, *[t.lower() for t in tokens]]
    return _finish(_join_parts(" ".join(pretty), time_phrase))


def _verb_sentence(
    subject_key: str | None,
    subject_text: str | None,
    verb: str,
    objects: list[str],
    future: bool,
    time_phrase: str,
) -> str:
    _ing, base = _VERBS[verb]
    obj_text = _object_phrase(subject_key, verb, objects)

    if future:
        if subject_text:
            core = f"{subject_text} will {base}"
        else:
            core = base.capitalize()
        return _finish(_join_parts(core, obj_text, time_phrase))

    if subject_text:
        if verb in _STATIVE_VERBS:
            core = f"{subject_text} {base}"
        else:
            be = _be_form(subject_key or "I")
            core = f"{subject_text} {be} {_ing}"
        return _finish(_join_parts(core, obj_text, time_phrase))

    # No subject: imperative (GO HOME → Go home.)
    core = base.capitalize()
    return _finish(_join_parts(core, obj_text, time_phrase))


def _object_phrase(
    subject_key: str | None,
    verb: str,
    objects: list[str],
) -> str:
    if not objects:
        return ""

    rendered: list[str] = []
    for obj in objects:
        if obj in _OBJECT_PRONOUNS:
            rendered.append(_OBJECT_PRONOUNS[obj])
        elif obj in _PEOPLE_NOUNS and subject_key == "I":
            rendered.append(f"my {_PEOPLE_NOUNS[obj]}")
        elif obj in _PEOPLE_NOUNS:
            rendered.append(f"the {_PEOPLE_NOUNS[obj]}")
        else:
            rendered.append(obj.lower())

    phrase = " ".join(rendered)

    if (
        verb in _MOTION_VERBS
        and objects
        and objects[0] not in _NO_PREPOSITION
        and objects[0] not in _OBJECT_PRONOUNS
    ):
        phrase = f"to {phrase}"

    return phrase


def _noun_as_subject(noun: str) -> str:
    if noun in _PEOPLE_NOUNS:
        return f"My {_PEOPLE_NOUNS[noun]}"
    return noun.capitalize()


def _be_form(subject_key: str) -> str:
    if subject_key == "I":
        return "am"
    if subject_key in {"HE", "SHE"}:
        return "is"
    return "are"


def _join_parts(*parts: str | None) -> str:
    return " ".join(part for part in parts if part).strip()


def _finish(sentence: str) -> str:
    sentence = " ".join(sentence.split())
    if not sentence:
        return ""
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


# ── LLM providers ────────────────────────────────────────────

def _user_prompt(words: Sequence[str]) -> str:
    glosses = " ".join(words)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Examples:\n{_FEW_SHOT_EXAMPLES}\n\n"
        f"ISL: {glosses}\n"
        f"English:"
    )


def _clean_llm_output(text: str) -> str:
    result = text.strip().strip("\"'`")
    result = " ".join(result.split())
    if not result:
        return ""
    # Keep only the first line in case the model adds extra text.
    result = result.splitlines()[0].strip()
    if result.lower().startswith("english:"):
        result = result.split(":", 1)[1].strip()
    return result


def _generate_with_gemini(words: Sequence[str], api_key: str) -> str:
    """Generate a sentence using the Google Gemini REST API."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": _user_prompt(words)}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 60,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        candidate = res_data.get("candidates", [{}])[0]
        content = (
            candidate.get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return _clean_llm_output(content)


def _generate_with_openai(words: Sequence[str], api_key: str) -> str:
    """Generate a sentence using the OpenAI Chat Completions REST API."""
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(words)},
        ],
        "temperature": 0.2,
        "max_tokens": 60,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        msg = res_data["choices"][0]["message"]["content"]
        return _clean_llm_output(msg)


def _generate_with_groq(words: Sequence[str], api_key: str) -> str:
    """Generate a sentence using Groq's OpenAI-compatible REST API."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        # A current, production Groq model suitable for short, fast completions.
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(words)},
        ],
        "temperature": 0.2,
        "max_tokens": 60,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        msg = res_data["choices"][0]["message"]["content"]
        return _clean_llm_output(msg)


# ── Offline self-check ───────────────────────────────────────

if __name__ == "__main__":
    os.environ["LLM_PROVIDER"] = "fallback"

    cases = [
        (["I", "GO", "SCHOOL", "TOMORROW"], "I am going to school tomorrow."),
        (["I", "HELP", "FRIEND", "TODAY"], "I am helping my friend today."),
        (["I", "GO", "HOME"], "I am going home."),
        (["I", "HAPPY"], "I am happy."),
        (["I", "SAD"], "I am sad."),
        (["HELLO"], "Hello."),
        (["GO", "HOME"], "Go home."),
        (["THANK", "YOU"], "Thank you."),
        (["FRIEND", "HELP", "ME"], "My friend is helping me."),
        (["I", "WILL", "GO", "SCHOOL", "TOMORROW"], "I will go to school tomorrow."),
    ]

    failed = 0
    for glosses, expected in cases:
        got = generate_sentence(glosses)
        status = "OK" if got == expected else "FAIL"
        if status == "FAIL":
            failed += 1
        print(f"{status}: {' '.join(glosses)}")
        print(f"     -> {got}")
        if status == "FAIL":
            print(f"     expected: {expected}")

    if failed:
        raise SystemExit(f"{failed} fallback case(s) failed.")
    print("All fallback cases passed.")
