#!/bin/bash
# the waiting queue; every step resumes from its own cache, so after a restart just run this again
cd "$(dirname "$0")"
python3 run_side_long.py 30 4 > side_long_30.out 2>&1
python3 run_tier2.py 100 4 > tier2_100.out 2>&1
echo QUEUE DONE
