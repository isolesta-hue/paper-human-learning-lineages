"""Speed probe (2026-10-12, user question: why is the D22 tale run so slow?). Read-only: profiles the first N seconds of one
life (seed 0) and prints the hottest functions; no cache is written (calls RL.life directly, not RL.one).
What this assumes: the first N seconds are representative of where the time goes; seed 0."""
import cProfile, pstats, signal, sys, io
import run_long as RL

cond, secs = sys.argv[1], int(sys.argv[2])
class Stop(Exception): pass
def h(*a): raise Stop
signal.signal(signal.SIGALRM, h); signal.alarm(secs)
pr = cProfile.Profile(); pr.enable()
try:
    RL.life(cond, 0)
except Stop:
    pass
pr.disable()
s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats("tottime").print_stats(25)
print(cond); print(s.getvalue()[:6000])
