"""Case definitions of the numerical-convergence study (see README.md).

Part: truncated_cone-s2026-0000 - a benchmark test part (design.csv of
benchmarks/springback/stage_n18): top radius 7.32 mm, depth 3.09 mm, wall
38.6 deg, top fillet 1.08 mm, bottom fillet 2.08 mm, footprint 15.4 mm -
formed as commanded (the uncompensated run). Its commanded height map is taken
from the benchmark's run cache (deck hash 3c6e05ba...).
"""
from pathlib import Path

PART_ID = "truncated_cone-s2026-0000"
PART_RUN = Path("/home/user/wt/bench/benchmarks/springback/work/runs/runs/3c/"
                "3c6e05ba7b508ae158f25c53446bbff3020340ee70ad512751f2d5cbed58660b")
# second check part (added on resume): truncated_cone-s2026-0001, the steeper
# test cone (wall 54.0 deg, depth 2.63 mm, top radius 8.14 mm, top fillet
# 2.33 mm, bottom fillet 1.49 mm, footprint 18.7 mm), uncompensated run.
# Its cases carry the prefix "c1_" and part="c1".
PARTS = {
    "t0": PART_RUN,
    "c1": Path("/home/user/wt/bench/benchmarks/springback/work/runs/runs/8c/"
               "8cf2f954fd91076354a6eeb088a757b992721815067ebcb0cc545534114b7941"),
}



def case(h=2.0, L=2, form="std", tp=0, travel=1.0, pen=10.0, explicit=None, **kw):
    d = dict(h=h, L=L, form=form, tp=tp, travel=travel, pen=pen)
    if explicit:
        d["explicit"] = explicit
    d.update(kw)
    return d


CASES = {
    # --- the benchmark setup, re-run with this build -------------------------
    "h2_L2_std": case(),
    # --- in-plane size ---------------------------------------------------------
    "h1_L2_std": case(h=1.0),
    "h0.625_L2_std": case(h=0.625),
    "h0.5_L2_std": case(h=0.5),
    # --- layers ----------------------------------------------------------------
    "h2_L4_std": case(L=4),
    "h1_L4_std": case(h=1.0, L=4),
    # --- element technology / rule through the thickness ---------------------
    "h2_L2_im": case(form="im"),
    "h2_L1_im_tp5": case(L=1, form="im", tp=5),
    "h2_L2_std_tp5": case(tp=5),
    "h2_L1_im_tp7": case(L=1, form="im", tp=7),
    "h2_L2_im_tp3": case(L=2, form="im", tp=3),
    "h1_L2_im": case(h=1.0, form="im"),
    "h1_L1_im_tp5": case(h=1.0, L=1, form="im", tp=5),
    "h1_L1_im_tp7": case(h=1.0, L=1, form="im", tp=7),
    "h1_L2_im_tp3": case(h=1.0, L=2, form="im", tp=3),
    "h0.833_L1_im_tp5": case(h=40.0 / 48, L=1, form="im", tp=5),
    "h0.833_L1_im_tp5_p3": case(h=40.0 / 48, L=1, form="im", tp=5, pen=3.0),
    "h0.625_L1_im_tp7": case(h=0.625, L=1, form="im", tp=7),
    "h0.5_L1_im_tp7": case(h=0.5, L=1, form="im", tp=7),
    "h0.625_L2_im_tp3": case(h=0.625, L=2, form="im", tp=3),
    # --- increment size, penalty ----------------------------------------------
    "h2_L2_std_t0.5": case(travel=0.5),
    "h1_L2_std_t0.5": case(h=1.0, travel=0.5),
    "h2_L2_std_p30": case(pen=30.0),
    "h2_L2_std_p3": case(pen=3.0),
    "h1_L2_std_p30": case(h=1.0, pen=30.0),
    "h2_L2_std_tol1e-8": case(solver_extra={"newton": {"residual_tolerance": 1e-8,
                                                       "displacement_tolerance": 1e-8}}),
    "h2_L2_std_rel40": case(release_increments=40),
    # --- explicit (standard Hex8 only: the dedicated kernel) ------------------
    # form_explicit was removed in the cleanup; these decks run only on a build
    # of commit 7d571e6 (their committed outputs are the record).
    "h1_L2_std_x1": case(h=1.0, explicit={"speed": 1.0, "dt": 1.0e-6, "sc": 0.5}),
    "h1_L2_std_x05": case(h=1.0, explicit={"speed": 0.5, "dt": 1.0e-6, "sc": 0.5}),
    "h2_L2_std_x05": case(explicit={"speed": 0.5, "dt": 1.0e-6, "sc": 0.5}),
    "h0.625_L2_std_x05": case(h=0.625, explicit={"speed": 0.5, "dt": 1.0e-6, "sc": 0.5}),
    "h0.5_L2_std_x05": case(h=0.5, explicit={"speed": 0.5, "dt": 1.0e-6, "sc": 0.5}),
    "h0.5_L4_std_x05": case(h=0.5, L=4, explicit={"speed": 0.5, "dt": 1.0e-6, "sc": 0.5}),
}

# --- added on resume -----------------------------------------------------------
CASES["h0.714_L1_im_tp5_p3"] = case(h=40.0 / 56, L=1, form="im", tp=5, pen=3.0)
CASES["h0.714_L1_im_tp5"] = case(h=40.0 / 56, L=1, form="im", tp=5)
for _n, _c in (("h2_L2_std", case()),
               ("h2_L1_im_tp5", case(L=1, form="im", tp=5)),
               ("h1_L1_im_tp5", case(h=1.0, L=1, form="im", tp=5)),
               ("h0.833_L1_im_tp5", case(h=40.0 / 48, L=1, form="im", tp=5)),
               ("h0.833_L1_im_tp5_p3", case(h=40.0 / 48, L=1, form="im", tp=5, pen=3.0)),
               ("h0.625_L1_im_tp5", case(h=0.625, L=1, form="im", tp=5))):
    CASES["c1_" + _n] = dict(_c, part="c1")
