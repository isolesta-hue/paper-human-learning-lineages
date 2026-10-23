"""Finding 102, the glyph lives first (the text lives with attention are slow): fills the cache for run_worldeye.py."""
from multiprocessing import Pool

import run_long as RL

if __name__ == "__main__":
    with Pool(4) as pool:
        pool.map(RL.one, [(sd, "both@ascii+fix-routeknobs+attend+worldeye") for sd in range(30)], chunksize=1)
    print("done")
