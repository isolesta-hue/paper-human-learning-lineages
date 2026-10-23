#!/bin/bash
# the waiting queue; every step resumes from its own per-life cache, so after a restart or a time-limit cut just run it again
cd "$(dirname "$0")"
python3 run_twoeyes.py 30 4 > twoeyes_30.out 2>&1 < /dev/null
python3 run_mode.py 30 4 > mode_30.out 2>&1 < /dev/null
python3 run_probe.py 10 4 > probe_10.out 2>&1 < /dev/null
python3 run_probe2.py 10 4 > probe2_10.out 2>&1 < /dev/null
python3 run_probe2.py 10 4 +attend glyph > attend_probe_10.out 2>&1 < /dev/null
python3 run_worldeye_glyph.py > worldeye_glyph.out 2>&1 < /dev/null
python3 run_rediscover.py 30 4 > rediscover_30.out 2>&1 < /dev/null
python3 run_bundle.py 30 4 > bundle_30.out 2>&1 < /dev/null
python3 run_rediscover100.py 100 4 > rediscover_100.out 2>&1 < /dev/null
python3 run_plastic.py 30 4 > plastic_30.out 2>&1 < /dev/null
python3 run_plastic2.py 30 4 > plastic2_30.out 2>&1 < /dev/null
python3 run_plastic3.py 30 4 > plastic3_30.out 2>&1 < /dev/null
python3 run_forget.py 30 4 > forget_30.out 2>&1 < /dev/null
python3 run_open.py 30 4 > open_30.out 2>&1 < /dev/null
python3 run_attend.py 30 4 > attend_30.out 2>&1 < /dev/null
python3 run_open.py 100 4 > open_100.out 2>&1 < /dev/null
python3 run_squeeze.py 10 4 > squeeze_10.out 2>&1 < /dev/null
python3 run_errors.py 10 4 > errors_10.out 2>&1 < /dev/null
python3 run_ladder1.py 30 4 > ladder1_30.out 2>&1 < /dev/null
python3 run_whatif.py 10 4 > whatif_10.out 2>&1 < /dev/null
python3 run_whatif2.py 10 4 > whatif2_10.out 2>&1 < /dev/null
python3 run_ladder1.py 100 4 > ladder1_100.out 2>&1 < /dev/null
python3 run_tune.py 30 4 > tune_30.out 2>&1 < /dev/null
python3 run_ladder2.py 30 4 > ladder2_30.out 2>&1 < /dev/null
python3 run_tune.py 100 4 plain > tune_100.out 2>&1 < /dev/null
python3 run_tunex.py 30 4 > tunex_30.out 2>&1 < /dev/null
python3 run_whatif3.py 10 4 > whatif3_10.out 2>&1 < /dev/null
python3 run_ladder2.py 100 4 > ladder2_100.out 2>&1 < /dev/null
python3 run_graded.py 30 4 > graded_30.out 2>&1 < /dev/null
python3 run_wm.py 30 4 > wm_30.out 2>&1 < /dev/null
python3 run_worldeye.py 30 4 > worldeye_30.out 2>&1 < /dev/null
python3 run_default.py 100 4 > default_100.out 2>&1 < /dev/null
python3 run_wmB.py 30 4 > wmB_30.out 2>&1 < /dev/null
python3 run_sideknobs.py 100 4 > sideknobs_100.out 2>&1 < /dev/null
python3 run_routeknobs.py 100 4 > routeknobs_100.out 2>&1 < /dev/null
python3 run_kor.py 30 4 > kor_30.out 2>&1 < /dev/null
python3 run_multi.py 30 4 > multi_30.out 2>&1 < /dev/null
python3 run_rev.py 30 4 > rev_30.out 2>&1 < /dev/null
python3 run_speak.py 30 4 > speak_30.out 2>&1 < /dev/null
python3 run_props.py 30 4 > props_30.out 2>&1 < /dev/null
python3 run_tale.py 30 4 > tale_30.out 2>&1 < /dev/null
python3 run_known.py 30 4 > known_30.out 2>&1 < /dev/null
python3 run_curious.py 30 4 > curious_30.out 2>&1 < /dev/null
python3 run_wmtest.py 30 4 > wmtest_30.out 2>&1 < /dev/null
python3 run_grav.py 10 4 > grav_10.out 2>&1 < /dev/null
python3 run_frame.py 10 4 > frame_10.out 2>&1 < /dev/null
python3 run_contain.py 10 4 > contain_10.out 2>&1 < /dev/null
python3 run_count.py 10 4 > count_10.out 2>&1 < /dev/null
python3 run_succ.py 10 4 > succ_10.out 2>&1 < /dev/null
python3 run_kor100.py 100 4 > kor_100.out 2>&1 < /dev/null
python3 run_numline.py 10 4 > numline_10.out 2>&1 < /dev/null
python3 run_views.py 10 4 > views_10.out 2>&1 < /dev/null
python3 run_viewpt.py 10 4 > viewpt_10.out 2>&1 < /dev/null
python3 run_goal.py 30 4 > goal_30.out 2>&1 < /dev/null
python3 run_multi50.py 100 4 > multi_100.out 2>&1 < /dev/null
python3 run_numline.py 30 4 > numline_30.out 2>&1 < /dev/null
echo QUEUE DONE
