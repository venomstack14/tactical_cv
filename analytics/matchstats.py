"""Ball-based match stats (possession / passes / turnovers) from tracked players + a detected ball. All values are PROXIES:
they need the detector to see the ball, and track-ID switches can split or merge a player's touches."""
import numpy as np
from collections import defaultdict

class MatchStats:
    def __init__(s, ctrl_m=3.0, margin_m=1.0, confirm_s=0.3, hold_s=0.5, pass_min_m=4.0, pass_max_gap_s=2.5):
        s.ctrl, s.margin, s.confirm, s.hold, s.pmin, s.pgap = ctrl_m, margin_m, confirm_s, hold_s, pass_min_m, pass_max_gap_s
        s.ball, s.ball_t, s.cand, s.cand_t0, s.owner, s.last_t = None, -1e9, None, 0.0, None, None
        s.last_touch = None                                     # (t, tid, team, xy) of the previous confirmed owner
        s.poss = defaultdict(float); s.passes = defaultdict(int); s.turnovers = defaultdict(int); s.pairs = defaultdict(int)
        s.frames = s.ball_frames = 0
    def update(s, t, ball_xy, players):
        """ball_xy: pitch metres or None. players: {track_id: (xy, team)} (team -1 = unknown/ref, ignored)."""
        dt = min(t - s.last_t, 0.5) if s.last_t is not None else 0.0; s.last_t = t; s.frames += 1
        if ball_xy is not None: s.ball, s.ball_t = np.asarray(ball_xy, float), t; s.ball_frames += 1
        if s.ball is None or t - s.ball_t > s.hold: s.cand = None; return        # ball not seen recently: no claim
        ds = sorted((float(np.hypot(*(np.asarray(xy) - s.ball))), tid, tm, xy) for tid, (xy, tm) in players.items() if tm >= 0)
        cand = None
        if ds and ds[0][0] <= s.ctrl:
            rival = next((d for d in ds[1:] if d[2] != ds[0][2]), None)           # nearest opponent of the closest player
            if rival is None or rival[0] - ds[0][0] >= s.margin: cand = ds[0]       # else contested -> no owner change
        if cand is None: s._credit(dt); return
        if s.cand is None or s.cand[1] != cand[1]: s.cand, s.cand_t0 = cand, t
        s.cand = cand
        if t - s.cand_t0 >= s.confirm and (s.owner is None or s.owner[0] != cand[1]): s._take(t, cand)
        s._credit(dt)
    def _credit(s, dt):
        if s.owner is not None: s.poss[s.owner[1]] += dt
    def _take(s, t, c):
        _, tid, team, xy = c
        if s.last_touch is not None:
            t0, pid, pteam, pxy = s.last_touch
            if t - t0 <= s.pgap:
                if pteam == team and np.hypot(*(np.asarray(xy) - pxy)) >= s.pmin: s.passes[team] += 1; s.pairs[(pid, tid)] += 1
                elif pteam != team: s.turnovers[pteam] += 1
        s.owner, s.last_touch = (tid, team), (t, tid, team, tuple(xy))
    def summary(s, our):
        tot = sum(s.poss.values()); opp = [k for k in (0, 1) if k != our][0]
        vis = s.ball_frames / max(s.frames, 1)
        return {"ball_visible_ratio": round(vis, 3), "controlled_seconds": round(tot, 1),
                "possession_pct": {"ours": round(100 * s.poss[our] / tot, 1) if tot else None, "opp": round(100 * s.poss[opp] / tot, 1) if tot else None},
                "passes": {"ours": s.passes[our], "opp": s.passes[opp]}, "turnovers": {"ours": s.turnovers[our], "opp": s.turnovers[opp]},
                "reliable": bool(vis >= 0.3 and tot >= 20), "note": "proxy: requires ball detections; unreliable when ball visibility < 30% or < 20 s controlled"}
