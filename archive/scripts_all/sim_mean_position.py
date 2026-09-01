"""Mean-Position simulator for the photon-CT board (pasted by user 2026-08-24).

Board rows = (Beam MAE, IDD, Strat, gamma, DVH, runtime) VALUES for the CURRENT top-10 + our entries.
Positions on the full board are deeper than top-10 (our 80.3s = rank 30), so for each metric we fit an
approximate value->position mapping from the pasted top-10 (positions in parens) and extrapolate for
worse values. Candidate entries are inserted, positions recomputed, Mean = (pos_beam + pos_idd +
pos_strat + pos_gamma + pos_dvh + 2*pos_runtime) / 7.

Usage: python scripts/sim_mean_position.py  (edit CANDIDATES below)
"""
# (value, known_position) pairs from the pasted 8/24 photon-CT board, per metric.
BEAM = [(0.0086,1),(0.0098,10),(0.0105,12),(0.0105,13),(0.0107,15),(0.0110,18),(0.0125,22),(0.0138,27),(0.0139,28),(0.0141,31)]
IDD  = [(0.0053,1),(0.0064,4),(0.0067,6),(0.0067,7),(0.0072,8),(0.0072,9),(0.0076,14),(0.0097,21),(0.0103,24),(0.0129,30)]
STRAT= [(0.0041,1),(0.0046,3),(0.0047,4),(0.0049,8),(0.0053,9),(0.0058,14),(0.0063,18),(0.0071,25),(0.0077,30),(0.0077,31)]
GAMMA= [(95.9974,1),(95.5115,4),(95.3257,5),(95.0669,8),(94.7723,13),(93.8248,17),(92.9550,19),(92.3169,24),(90.7783,27),(90.7526,28)]  # higher better
DVH  = [(0.2029,1),(0.2358,4),(0.2596,5),(0.2917,10),(0.2962,11),(0.3004,14),(0.3132,18),(0.3281,19),(0.3764,21),(0.3810,22)]
RT   = [(35.8553,4),(42.6697,8),(42.8395,9),(49.8684,10),(57.8533,13),(58.9689,14),(62.4678,19),(80.3151,30),(92.2038,38),(103.6138,46)]

def pos_of(value, table, higher_better=False):
    """Interpolate/extrapolate a board position for `value` from (value, pos) anchors."""
    t = sorted(table, key=lambda r: r[0], reverse=higher_better)   # best first
    if (value <= t[0][0]) != higher_better or value == t[0][0]:
        # better than or equal to the best anchor
        if higher_better and value >= t[0][0]: return max(t[0][1] - (value - t[0][0]) * 50, 1)
        if not higher_better and value <= t[0][0]: return max(t[0][1] - (t[0][0] - value) * 1000, 1)
    prev = t[0]
    for cur in t[1:]:
        between = (prev[0] <= value <= cur[0]) if not higher_better else (cur[0] <= value <= prev[0])
        if between:
            f = (value - prev[0]) / (cur[0] - prev[0] + 1e-12)
            return prev[1] + f * (cur[1] - prev[1])
        prev = cur
    # worse than the worst anchor: linear extrapolate with the last segment's slope
    a, b = t[-2], t[-1]
    slope = (b[1] - a[1]) / (b[0] - a[0] + 1e-12)
    return b[1] + slope * (value - b[0])

def mean_position(beam, idd, strat, gamma, dvh, rt):
    p = [pos_of(beam, BEAM), pos_of(idd, IDD), pos_of(strat, STRAT),
         pos_of(gamma, GAMMA, higher_better=True), pos_of(dvh, DVH), pos_of(rt, RT), pos_of(rt, RT)]
    return sum(p) / 7.0, [round(x,1) for x in p]

# ---- sanity: reproduce known entries ----
KNOWN = {
    "ix 1st (10.4)":      (0.0105, 0.0097, 0.0053, 94.7723, 0.2029, 42.6697),
    "TS_UKE 2nd (11.1)":  (0.0098, 0.0076, 0.0046, 95.0669, 0.2596, 62.4678),
    "OURS 5th (15.7)":    (0.0107, 0.0072, 0.0049, 95.3257, 0.3004, 80.3151),
}
print("== sanity (should match board Mean) ==")
for k, v in KNOWN.items():
    m, p = mean_position(*v)
    print(f"  {k}: sim Mean {m:.1f}  positions {p}")

# ---- candidates: edit as internal numbers arrive ----
# Format: (label, beam, idd, strat, gamma, dvh, runtime_s)
CANDIDATES = [
    # what-if: our current accuracy, faster runtime only
    ("ours @60s",  0.0107, 0.0072, 0.0049, 95.3257, 0.3004, 60.0),
    ("ours @50s",  0.0107, 0.0072, 0.0049, 95.3257, 0.3004, 50.0),
    ("ours @43s",  0.0107, 0.0072, 0.0049, 95.3257, 0.3004, 43.0),
    # what-if: DVH fixed to ix level, same runtime
    ("ours dvh.21",0.0107, 0.0072, 0.0049, 95.3257, 0.2100, 80.3),
    # what-if: both
    ("ours @60s+dvh.24", 0.0107, 0.0072, 0.0049, 95.3257, 0.2400, 60.0),
    ("ours @50s+dvh.24", 0.0107, 0.0072, 0.0049, 95.3257, 0.2400, 50.0),
]
print("\n== candidates ==")
for row in CANDIDATES:
    m, p = mean_position(*row[1:])
    print(f"  {row[0]:22s}: Mean {m:.1f}  positions {p}   {'*** BEATS 10.4 ***' if m < 10.4 else ''}")
