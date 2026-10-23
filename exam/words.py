"""Closed-class words (exam_spec/closed_words.json, closed-v1) and the pseudo-words of an edition (open class: people,
things, verbs; EXAM_DESIGN.md 2). Korean-like pseudo-words are composed of Hangul syllables, English-like ones of letters;
both from fixed small inventories so they are pronounceable; none equals a closed word or another pseudo-word."""
import json
import os

from .common import rng

HERE = os.path.dirname(os.path.abspath(__file__))
CLOSED = json.load(open(os.path.join(HERE, "..", "exam_spec", "closed_words.json"), encoding="utf-8"))

KO_INITIAL = [0, 2, 3, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18]   # ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅅ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ
KO_MEDIAL = [0, 4, 8, 13, 18, 20, 1, 5]                           # ㅏ ㅓ ㅗ ㅜ ㅡ ㅣ ㅐ ㅔ
KO_FINAL = [0, 0, 0, 4, 8, 16, 21, 1]                             # (none x3) ㄴ ㄹ ㅁ ㅇ ㄱ
EN_ONSET = list("bdfgklmnprstvz") + ["bl", "br", "dr", "gl", "kl", "pr", "sk", "sl", "sn", "st", "tr"]
EN_VOWEL = ["a", "e", "i", "o", "u", "oo", "ee", "ai"]
EN_CODA = ["", "", "b", "d", "g", "k", "l", "m", "n", "p", "t", "x", "sk", "nd"]


def closed_tokens():
    out = set()
    for e in CLOSED["numbers"] + CLOSED["spatial"] + CLOSED["wh"]:
        for lang in ("ko", "en"):
            v = e[lang]
            out.update(v if isinstance(v, list) else [v])
    for p in CLOSED["ko_particles"]:
        out.update(v for k, v in p.items() if k != "role")
    for f in CLOSED["en_function"]:
        out.update(f.get("forms", [f.get("form")]))
    for f in CLOSED["ko_function"]:
        out.add(f["form"])
    return {x for x in out if x}


def ko_syllable(i, m, f):
    return chr(0xAC00 + (i * 21 + m) * 28 + f)


def has_final(word):
    """the last syllable has a final consonant (받침)."""
    c = ord(word[-1]) - 0xAC00
    return 0 <= c < 11172 and c % 28 != 0


def final_is_rieul(word):
    c = ord(word[-1]) - 0xAC00
    return 0 <= c < 11172 and c % 28 == 8


def particle(word, role):
    """the Korean particle for a (pseudo-)word by its final sound (closed-v1 ko_particles)."""
    p = {x["role"].split(" ")[0]: x for x in CLOSED["ko_particles"]}
    if role == "subject":
        return p["subject"]["after_consonant"] if has_final(word) else p["subject"]["after_vowel"]
    if role == "object":
        return p["object"]["after_consonant"] if has_final(word) else p["object"]["after_vowel"]
    if role == "direction":
        d = p["direction"]
        return d["after_vowel_or_rieul"] if (not has_final(word) or final_is_rieul(word)) else d["after_consonant"]
    if role == "location":
        return "에"
    if role == "source":
        return "에서"
    raise ValueError(role)


class Lexicon:
    """an edition-life's pseudo-words: one draw stream per kind of word, no repeats, no closed words."""

    def __init__(self, seed, life):
        self.r = rng(seed, "lexicon", life)
        self.used = set(closed_tokens())

    def _ko(self, n_syll, verb=False):
        while True:
            w = "".join(ko_syllable(int(self.r.choice(KO_INITIAL)), int(self.r.choice(KO_MEDIAL)),
                                    int(self.r.choice(KO_FINAL)) if (k < n_syll - 1 or not verb) else 0)
                        for k in range(n_syll))
            if verb:
                w = w + "다"
            elif w.endswith("다"):                     # a noun must not look like a verb
                continue
            if w not in self.used:
                self.used.add(w)
                return w

    def _en(self, n_syll):
        while True:
            w = "".join(str(self.r.choice(EN_ONSET)) + str(self.r.choice(EN_VOWEL)) for _ in range(n_syll - 1))
            w += str(self.r.choice(EN_ONSET)) + str(self.r.choice(EN_VOWEL)) + str(self.r.choice(EN_CODA))
            if w not in self.used and 3 <= len(w) <= 9:
                self.used.add(w)
                return w

    def name(self):
        """a person's name: one form used in both languages (names are not translated)."""
        return self._ko(2)

    def noun(self):
        return {"ko": self._ko(2), "en": self._en(int(self.r.integers(1, 3)))}

    def verb(self):
        return {"ko": self._ko(2, verb=True), "en": self._en(int(self.r.integers(1, 3)))}
