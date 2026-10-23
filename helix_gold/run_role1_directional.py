"""Directional connections for internal logic 2 (finding 31 follow-up).

Finding 31: with the symmetric PPMI connection seed, a new word did not bring
back the already-known word it always comes with ("dad" -> "mom"), because
PPMI discounts a word that appears in many sentences. People seem to use
directional links instead: "dad" (always heard with "mom") points strongly
to "mom", while "mom" (heard in many places) points only weakly to "dad".

internal logic 2, directional version (dhebb): conn[a, b] = P(b | a), the
share of the sentences holding a that also hold b (run_connection_seed.py,
DirectionalHebbian). Same test as finding 31 (run_role1_anchor.py): cross
pairs of one new + one already-known word in sessions 2 and 3.
Conditions: none / hebb (PPMI) / dhebb / vec+hebb / vec+dhebb / vec+bert.
"""
from run_role1_anchor import main

CONDITIONS = ["none", "hebb", "dhebb", "vec+hebb", "vec+dhebb", "vec+bert"]

if __name__ == "__main__":
    main(CONDITIONS)
