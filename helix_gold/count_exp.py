"""Experience ledger probe (D71): what one Dami life hears, per world (read only, builds the world and counts)."""
import sys, numpy as np
sys.argv = sys.argv[:1]
import run_long as RL
from run_schema import PER_SESSION, EPOCHS
import run_integrated as RI
worlds = ["ascii-ten", "ascii-sp", "ascii-kor", "ascii-c8", "ascii-when", "ascii-bi", "ascii-count", "ascii-tale", "ascii-nw"]
print("EPOCHS", EPOCHS, "PER_SESSION", PER_SESSION, "HOLD", RI.HOLD)
for name in worlds:
    try:
        w = RL.build_world(name, 0)
    except Exception as e:
        print(name, "ERR", e); continue
    heard = [s for c in range(w.n_cat) for _, s, _ in w.categories[c]["heard"]]
    toks = [len(s) if hasattr(s, "__len__") else 1 for s in heard]
    vocab = set(t for s in heard for t in (s if hasattr(s, "__iter__") else [s]))
    extra = {k: len(v) for k, v in vars(w).items() if isinstance(v, (list, dict)) and k not in ("categories",)}
    print(f"{name:12} rooms {w.n_cat:3} sessions {int(np.ceil(w.n_cat/PER_SESSION)):3} heard {len(heard):5} "
          f"kept {int(len(heard)*(1-RI.HOLD)):5} words {sum(toks):6} vocab {len(vocab):4} ex {heard[0]!r:.60}")
