#!/bin/bash
# the waiting queue; every step resumes from its own per-life cache, so after a restart or a time-limit cut just run it again
cd "$(dirname "$0")"
# 10-13 02:40Z: restart after a container stop; finished lines (up to run_props, and the helper-finished numline/views/viewpt 10, goal 30) moved to run_detached_history_1013.sh so nothing finished is rerun or overwritten
# 10-13 04:00Z (user decision): the queue runs in PIECES that finish within the background-task limit, each piece one harness
# background job: RIE_DEADLINE=<epoch> bash run_detached.sh. No new life starts after the deadline (run_long.one); a runner cut
# by it ends with "PIECE DEADLINE" in its output and the piece stops there; a step whose output is already complete is
# skipped, so nothing finished is rerun or overwritten. W2 (track A, run_newword.py) right after D22 (user decision 10-13).
step() {    # step <out file> <command ...>
    local out=$1; shift
    if [ -s "$out" ] && ! grep -q "Traceback\|PIECE DEADLINE" "$out"; then return 0; fi
    if [ -n "$RIE_DEADLINE" ] && [ "$(date +%s)" -ge "$RIE_DEADLINE" ]; then echo "piece deadline before $out"; exit 0; fi
    "$@" > "$out" 2>&1 < /dev/null
    if grep -q "PIECE DEADLINE" "$out"; then echo "piece ended inside $out"; exit 0; fi
}
step tale_30.out python3 run_tale.py 30 4
step newword_10.out python3 run_newword.py 10 4
step known_30.out python3 run_known.py 30 4
step spatial_10.out python3 run_spatial.py 10 4
step when_10.out python3 run_when.py 10 4
step d25_10.out python3 run_d25.py 10 4
step curious_30.out python3 run_curious.py 30 4
step wmtest_30.out python3 run_wmtest.py 30 4
step grav_10.out python3 run_grav.py 10 4
step frame_10.out python3 run_frame.py 10 4
step contain_10.out python3 run_contain.py 10 4
step count_10.out python3 run_count.py 10 4
step succ_10.out python3 run_succ.py 10 4
step kor_100.out python3 run_kor100.py 100 4
step multi_100.out python3 run_multi50.py 100 4
step numline_30.out python3 run_numline.py 30 4
echo QUEUE DONE
