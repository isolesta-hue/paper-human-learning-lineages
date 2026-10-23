"""Run one life with the hidden-state audit on (no cache read or written). argv: condition, seed, out path, stop session."""
import os, sys, torch
os.environ["VOY_STATEDIFF"] = sys.argv[3]
os.environ["VOY_STATEDIFF_STOP"] = sys.argv[4]
import run_long as RL
torch.set_num_threads(1)
RL.life(sys.argv[1], int(sys.argv[2]))
print("done", sys.argv[3])
