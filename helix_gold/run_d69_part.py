"""D69 compute share for the compute helper (a wrapper with NO reading, user decision 10-17 Q11 -> the helper stops
wmtest_30 and takes part of D69): the SAME tasks as run_d69.py (its list, its condition names, its cache), taken from
the END of the list, so the main queue (which runs from the front) and the helper meet in the middle; a life already
cached is skipped (run_long.one). The pre-registered reading stays run_d69.py's, run in the main session on all 60.
python3 run_d69_part.py N_SEEDS WORKERS [HOW_MANY]"""
import sys

import run_long as RL
import run_d69 as D
import run_w7pilot as P

if __name__ == "__main__":
    n_seeds, workers = int(sys.argv[1]), int(sys.argv[2])
    k = int(sys.argv[3]) if len(sys.argv) > 3 else None
    seeds = list(range(n_seeds))
    tasks = [(sd, D.COND[x](w_)) for w_ in P.FIT for x in ("F", "FG", "AG") for sd in seeds][::-1][:k]
    done = RL.map_lives(tasks, workers)
    print(f"D69 part: {sum(r is not None for r in done)} / {len(tasks)} lives cached (from the end of run_d69's list)")
