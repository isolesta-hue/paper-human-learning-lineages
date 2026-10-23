#!/bin/bash
# one diagnosis piece: each life in its own process (an out-of-memory kill of one life cannot stop the others), a deadline,
# and a memory sampler (largest diagnosis process, every 10 s)
SP=/tmp/claude-0/-home-user/23cc98b2-a804-5343-bef4-6479ad5f8346/scratchpad
N=$1
cd $SP/repo
DL=$(( $(date +%s) + 105*60 ))
echo "piece $N start $(date -u +%FT%TZ) deadline $(date -u -d @$DL +%TZ) (one process per life)" >> $SP/diag_joined/pieces.log
( while true; do ps -eo rss,args --sort=-rss | grep "diag_joine[d]" | awk -v t="$(date -u +%T)" '{s+=$1; if ($1>m) m=$1} END {printf "%s max %.2f GB total %.2f GB\n", t, m/1048576, s/1048576}' >> $SP/diag_joined/rss_piece$N.log; sleep 10; done ) &
S=$!
for L in life_001 life_007 life_018; do
  if [ ! -f $SP/diag_joined/$L.json ]; then
    python3 diag_joined.py /home/user/shared-curriculum/exam/v1 $SP/exam_v1_dami/answers $SP/diag_joined $L 1 --deadline $DL > $SP/diag_joined/piece${N}_$L.log 2>&1 &
  fi
done
wait $(jobs -p | grep -v "^$S$")
kill $S
echo "piece $N end $(date -u +%FT%TZ)" >> $SP/diag_joined/pieces.log
