cd /tmp/claude-0/-home-user/23cc98b2-a804-5343-bef4-6479ad5f8346/scratchpad/rie
for w in reg G Gr Gn body; do python3 check_goal.py $w > check_goal_$w.out 2>&1 < /dev/null; done
echo CHECKS DONE > check_goal_done.out
