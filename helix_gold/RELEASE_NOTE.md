# Helix Gold — release note

This folder is a copy of Helix Gold's working repository (2026-10-23), with what the paper's numbers come from:
- the code (`*.py`), the runners (`run_*.py`, usually `python3 run_xxx.py <seeds> [workers]`) and the shell scripts that queued them;
- every result file (`*.out`), the pre-registrations and design notes (`*_DESIGN.md`, `DIAG_JOINED.md`, …), the score table (`SCORE_TABLE.md`), the record of what humans gave the learner (`HUMAN_SHARE.md`);
- the research record (`README.md`, in Korean): every numbered finding ("finding N" in the paper), with its runner and output file;
- the ledger (`LEDGER.md`): every promise, decision and open idea (the "D" and "Q" numbers in the paper's Appendix F);
- the v1 grading outputs (`exam_grade_v1/`). The v1 answers are in `../answers/v1/helix_gold/`.

Left out: the cached lives (pickles, several hundred MB — every number is in the `*.out` files and can be recomputed by the runners), the mascot images, and files that only ran the working sessions (instructions to the assistant, session hand-offs, the sibling-reading script, the dashboard).

Names: the code and records use the working names — Voyager (= Helix Gold, its session "Dami"), Pioneer (= Helix Teal, "Ara"), New Horizons (= Helix Blue, "Sia").
