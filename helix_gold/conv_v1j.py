"""conv-v1j -- DIAGNOSIS ONLY (finding 166, next (a); the user's decision 10-21 "응, 진단시험을 하자."). Not an exam
converter: edition v1's keys are public since 10-21 and the v1 answers stay as graded (conv-v1).

conv-v1 with ONE change: an exposure line becomes ONE heard unit -- its sentence moment and its scene-line moments joined by
the boundary element, in conv-v1's own order (the sentence, then one [picture, at] per scene line), the same joining
conv-v1 already uses for an item's context (Query.sequence). conv-v1 hears them as separate moments, and Dami's body pairs
elements only inside a moment, so a name and the picture it names never met (finding 166 (1)). Everything else is
conv-v1's own code, unchanged: the elements and their features, the items, the candidates, the answers.
Fixed before any diagnosis life runs (DIAG_JOINED.md)."""
import conv_v1 as CV

VERSION = "conv-v1j"
_convert = CV.convert_life                       # conv-v1's own; kept before a runner swaps CV.convert_life


def join_heard(heard, boundary):
    """conv-v1's heard moments of one session ("conv sentence", then its "conv line"s, per exposure line, in order) ->
    one ("conv record", sentence + [boundary + line]...) unit per exposure line."""
    out = []
    for kind_, toks, _ in heard:
        if kind_ == "conv sentence":
            out.append(("conv record", list(toks), []))
        else:
            assert kind_ == "conv line" and out, kind_
            out[-1][1].extend([boundary] + list(toks))
    return out


def convert_life(life_dir, tag=None):
    w, qs = _convert(life_dir, tag) if tag is not None else _convert(life_dir)
    for c in w.sessions:
        w.categories[c]["heard"] = join_heard(w.categories[c]["heard"], w.boundary)
    w.version = VERSION
    return w, qs


def rebuild_records(w):
    """the exposure lines back from the joined units: [{session, sentence, scene: {lines: [{id, cells}]}}] in order."""
    out = []
    for c in w.sessions:
        for _, toks, _ in w.categories[c]["heard"]:
            parts, cur = [], []
            for t in toks:
                if t == w.boundary:
                    parts.append(cur)
                    cur = []
                else:
                    cur.append(t)
            parts.append(cur)
            out.append({"session": c - 1, "sentence": [w.word_of[t] for t in parts[0]],
                        "scene": {"lines": [{"id": w.id_of[p[0]], "cells": [list(w.cell_of[t]) for t in p[1:]]}
                                            for p in parts[1:]]}})
    return out
