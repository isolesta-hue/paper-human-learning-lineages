#!/bin/bash
# wait for finding 97's output, then restart the queue so it reads the updated list (finding 98's run next)
cd "$(dirname "$0")"
until grep -q "^K4" twoeyes_30.out 2>/dev/null; do sleep 30; done
sleep 5
for p in $(ps -eo pid,cmd | grep -E "python3 run_|bash ./run_detached|run_detached.sh$" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 3
echo relaunch
exec ./run_detached.sh
