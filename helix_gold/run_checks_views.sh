cd /tmp/claude-0/-home-user/23cc98b2-a804-5343-bef4-6479ad5f8346/scratchpad/rie
for w in reg V Vr Vn body; do python3 check_views.py $w > check_views_$w.out 2>&1 < /dev/null; done
echo CHECKS DONE > check_views_done.out
