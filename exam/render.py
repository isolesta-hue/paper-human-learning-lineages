"""Sentences in two languages (exam_spec/GENERATOR.md 3): one event, Korean-like (particles chosen by the final sound,
number after the noun, verb last) or English-like (subject-verb-object + prepositions, no articles). Tokens are words;
particles and function words are separate tokens."""
from .words import CLOSED, particle

NUM = {e["n"]: e for e in CLOSED["numbers"]}
SPATIAL = {e["en"]: e for e in CLOSED["spatial"]}


def _w(lex, k, lang):
    """the word of an entity: a person's name (one form) or its kind's noun in the language."""
    return lex["names"][k] if k in lex["names"] else lex["nouns"][lex["kind_of"][k]][lang]


def sentence(ev, lex, lang):
    t = ev["type"]
    if lang == "ko":
        if t == "exist":
            a = _w(lex, ev["a"], "ko")
            out = [a, particle(a, "subject")]
            if ev.get("rel"):
                b = _w(lex, ev["b"], "ko")
                out += [b, SPATIAL[ev["rel"]]["ko"], particle(b, "location")]
            return out + ["있다"]
        if t == "count":
            noun = lex["nouns"][ev["kind"]]["ko"]
            num = NUM[ev["n"]]["ko"]
            out = [noun, num, particle(num, "subject")]
            if ev.get("rel"):
                b = _w(lex, ev["b"], "ko")
                out += [b, SPATIAL[ev["rel"]]["ko"], particle(b, "location")]
            return out + ["있다"]
        a = _w(lex, ev["c"], "ko")
        x = _w(lex, ev["t"], "ko")
        v = lex["verbs"][t]["ko"]
        if t == "go":
            return [a, particle(a, "subject"), x, particle(x, "direction"), v]
        if t == "push":
            return [a, particle(a, "subject"), x, particle(x, "object"), v]
        if t == "put":
            b = _w(lex, ev["b"], "ko")
            return [a, particle(a, "subject"), x, particle(x, "object"), b, SPATIAL[ev["rel"]]["ko"], particle(b, "location"), v]
        if t == "take":
            b = _w(lex, ev["b"], "ko")
            return [a, particle(a, "subject"), x, particle(x, "object"), b, particle(b, "source"), v]
    else:
        if t == "exist":
            out = ["there", "is", _w(lex, ev["a"], "en")]
            if ev.get("rel"):
                out += [SPATIAL[ev["rel"]]["en"], _w(lex, ev["b"], "en")]
            return out
        if t == "count":
            out = ["there", "is" if ev["n"] == 1 else "are", NUM[ev["n"]]["en"], lex["nouns"][ev["kind"]]["en"]]
            if ev.get("rel"):
                out += [SPATIAL[ev["rel"]]["en"], _w(lex, ev["b"], "en")]
            return out
        a = _w(lex, ev["c"], "en")
        x = _w(lex, ev["t"], "en")
        v = lex["verbs"][t]["en"]
        if t == "go":
            return [a, v, "to", x]
        if t == "push":
            return [a, v, x]
        if t == "put":
            return [a, v, x, SPATIAL[ev["rel"]]["en"], _w(lex, ev["b"], "en")]
        if t == "take":
            return [a, v, x, "from", _w(lex, ev["b"], "en")]
    raise ValueError(ev)
