"""Ledger A3 (2026-10-10): why does never-heard fall over the life? From the cached lives only (no new run): per session,
the room learned this session ("newest") vs the earlier rooms ("older"), and the items heard once ("once all") vs the pairs
heard only apart ("unseen all"). Proactive interference / a fan effect predicts the NEWEST rooms falling too; plain
forgetting predicts only the older ones falling."""
import pickle, sys, numpy as np
cond = sys.argv[1]
outs = [pickle.load(open(f"cache_long_v2/seed{s}_{cond}.pkl", "rb")) for s in range(30)]
print(cond, "(30 seeds; sessions 1..16; phase 2 from session 5, phase 3 = upside-down rooms from session 9-ish)")
for k in ("never-heard", "newest", "older", "once all", "unseen all"):
    m = np.nanmean(np.array([[st.get(k, np.nan) for st in o] for o in outs]), 0)
    print(f"  {k:11} " + " ".join(f"{v:.3f}" for v in m))
