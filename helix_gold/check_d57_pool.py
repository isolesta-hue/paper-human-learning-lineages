"""D57 check: map_lives returns results in task order, and a worker killed by a signal fails loudly AFTER the others finish."""
import os
import signal
import sys
import time

import run_long as RL


def fn(t):
    time.sleep(0.2 * (t % 3))
    if t == 4:
        os.kill(os.getpid(), signal.SIGKILL)     # as the kernel's out-of-memory kill
    return t * t


print("order:", RL.map_lives(list(range(4)), 2, fn=fn, peak=lambda t: 0.1) == [0, 1, 4, 9], flush=True)
try:
    RL.map_lives(list(range(7)), 3, fn=fn, peak=lambda t: 0.1)
    print("NOT RAISED")
except RuntimeError as e:
    print("raised:", e)
print("tmp left:", os.listdir(RL.CACHE + "_tmp"))
print("free GB now:", round(RL._mem_free_gb(), 1), "| W5 peak estimate:", RL.life_peak_gb((0, "both@ascii@ten+x")))
