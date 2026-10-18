"""Deterministic counter-tactical decision engine ("Juego de Posicion" principles) over a 5-second rolling window.

The analysed unit is the OPPONENT block (what we must beat) plus our own occupation of space. Everything is rule-based and
explainable: each recommendation carries the numbers that triggered it. Metrics only use players the camera currently sees,
so below `min_visible` opponents the engine reports 'low coverage' instead of inventing numbers (broadcast limitation).
Pitch convention: x along length, y across; `attack_dir=+1` means OUR team attacks toward +x (opponent goal at x=105)."""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from scipy.spatial import ConvexHull
from core import pitch

LANES = [13.6, 27.2, 40.8, 54.4]                  # 5 vertical channels: wide | half-space | centre | half-space | wide
@dataclass
class Snapshot:
    t: float
    ours: dict       # id -> (x, y) metres
    opp: dict
    ball: tuple | None = None
@dataclass
class Recommendation:
    priority: int    # 1 = act now, 2 = important, 3 = info
    tag: str
    message: str
    evidence: str
    def __str__(s): return f"[P{s.priority}] {s.message}"
@dataclass
class Thresholds:
    high_line: float = 40.0; low_block: float = 25.0; narrow: float = 38.0; wide: float = 55.0; compact_hi: float = 0.75; compact_lo: float = 0.40
    interline: float = 16.0; press_close_ms: float = 1.5; press_radius: float = 8.0; press_min: int = 2; hs_overload: int = 4
    hold: int = 3; release: int = 3              # hysteresis (consecutive evaluations) so advice does not flicker

class TacticsEngine:
    def __init__(s, attack_dir=+1, window_s=5.0, min_visible=7, th=None):
        s.dir, s.win, s.minv, s.th = attack_dir, window_s, min_visible, th or Thresholds(); s.buf = deque(); s.streak = {}; s.active = {}
    def push(s, snap):
        s.buf.append(snap)
        while s.buf and snap.t - s.buf[0].t > s.win: s.buf.popleft()
    # ----------------------------------------------------------------- geometry helpers
    def _depth(s, x): return np.abs((pitch.L if s.dir > 0 else 0.0) - np.asarray(x))        # distance from the opponent's own goal line
    def _outfield(s, pts):
        """Opponent outfield players: the deepest man is dropped as 'goalkeeper' if he is near the goal and enough others are visible."""
        a = np.array(list(pts.values()), float).reshape(-1, 2)
        if len(a) >= 8:
            k = s._depth(a[:, 0]).argmin()
            if s._depth(a[k, 0]) < 20: a = np.delete(a, k, 0)
        return a
    def _snap_metrics(s, sn):
        o = s._outfield(sn.opp)
        if len(o) < s.minv - 1: return None
        d = np.sort(s._depth(o[:, 0])); back = d[:min(4, len(d))].mean(); area = ConvexHull(o).volume if len(o) >= 3 else 0.0
        cuts = sorted(np.argsort(np.diff(d))[-2:] + 1) if len(d) >= 6 else []                    # split into 3 lines at the two largest gaps
        parts = np.split(d, cuts) if cuts else [d]; means = [float(p.mean()) for p in parts]
        lanes = np.bincount(np.digitize(o[:, 1], LANES), minlength=5); ours = np.array(list(sn.ours.values()), float).reshape(-1, 2)
        in_opp_half = ours[s._depth(ours[:, 0]) < 52.5] if len(ours) else ours
        return dict(line_height=float(back), block_depth=float(d.max() - d.min()), width=float(np.ptp(o[:, 1])), hull_area=float(area),
                    compactness=float(np.clip(1 - (area - 800) / 2400, 0, 1)), interline=float(means[1] - means[0]) if len(means) >= 2 else 0.0,
                    lanes=lanes, opp_hs=np.array([lanes[1], lanes[3]]), our_hs=np.array([(np.digitize(in_opp_half[:, 1], LANES) == 1).sum(), (np.digitize(in_opp_half[:, 1], LANES) == 3).sum()]) if len(in_opp_half) else np.zeros(2, int))
    def _pressing(s):
        """Opponents closing in on one of our players: >= press_min opponents within radius closing faster than press_close_ms (needs ~1 s of history)."""
        if len(s.buf) < 3: return 0, None
        now = s.buf[-1]; past = next((b for b in s.buf if now.t - b.t <= 1.2 and now.t - b.t >= 0.5), None)
        if past is None: return 0, None
        best, who = 0, None; targets = ({k: v for k, v in now.ours.items()} if now.ball is None else {k: v for k, v in now.ours.items() if np.hypot(v[0] - now.ball[0], v[1] - now.ball[1]) < 4})
        for oid, op in targets.items():
            if oid not in past.ours: continue
            n = 0
            for pid, pp in now.opp.items():
                if pid not in past.opp: continue
                d1 = np.hypot(pp[0] - op[0], pp[1] - op[1]); d0 = np.hypot(past.opp[pid][0] - past.ours[oid][0], past.opp[pid][1] - past.ours[oid][1])
                if d1 < s.th.press_radius and (d0 - d1) / (now.t - past.t) >= s.th.press_close_ms: n += 1
            if n > best: best, who = n, oid
        return best, who
    def metrics(s):
        ms = [m for m in (s._snap_metrics(b) for b in s.buf) if m is not None]
        if len(ms) < max(3, len(s.buf) // 3): return None
        med = lambda k: np.median([m[k] for m in ms], 0); out = {k: float(med(k)) for k in ("line_height", "block_depth", "width", "hull_area", "compactness", "interline")}
        out["opp_hs"], out["our_hs"], out["window_s"] = np.round(med("opp_hs")).astype(int), np.round(med("our_hs")).astype(int), float(s.buf[-1].t - s.buf[0].t)
        first = next((m for m in ms[:max(1, len(ms) // 4)]), ms[0]); out["line_shift"] = float(np.mean([m["line_height"] for m in ms[-max(1, len(ms) // 4):]]) - first["line_height"])
        out["press_n"], out["press_target"] = s._pressing(); return out
    def _side(s, lane_idx):  # lane 1 = low-y half-space, lane 3 = high-y; name it from OUR attacking view (facing +x, left = +y)
        return "left" if (lane_idx == 3) == (s.dir > 0) else "right"
    # ----------------------------------------------------------------- rules
    def evaluate(s, profiles=None):
        """-> (metrics|None, [Recommendation]) sorted by priority. profiles: {opp_track_id: PlayerProfile}."""
        m, t = s.metrics(), s.th
        if m is None: return None, [Recommendation(3, "LOW_COVERAGE", "Too few opponents in view - tactical read suppressed.", f"need >= {s.minv} visible opponents")]
        R = {}
        def rule(tag, cond, pr, msg, ev):
            if cond: R[tag] = Recommendation(pr, tag, msg, ev)
        rule("HIGH_LINE", m["line_height"] >= t.high_line, 1, "High defensive line: attack the space behind it - one forward pins the centre-backs, winger/#8 run in behind (tercer hombre), play the diagonal early.", f"line {m['line_height']:.0f} m from their goal")
        rule("LOW_BLOCK", m["line_height"] <= t.low_block, 1, "Deep block: do not force depth. Circulate with pauses to move the block, create superioridad in the half-spaces, cut-backs and shots from the edge; avoid crosses into a crowded box.", f"line {m['line_height']:.0f} m")
        rule("NARROW", m["width"] <= t.narrow, 2, "Narrow block: stretch with wingers on the touchline (amplitud) to fix the full-backs, then switch play (cambio de orientacion) to the isolated wide player.", f"width {m['width']:.0f} m")
        rule("STRETCHED", m["width"] >= t.wide and m["compactness"] < t.compact_lo, 2, "Opponent is stretched with gaps between channels: find the free man between the lines and combine through the half-spaces.", f"width {m['width']:.0f} m, compactness {m['compactness']:.2f}")
        rule("INTERLINE", m["interline"] >= t.interline, 2, "Large space between their back line and midfield line: receive on the half-turn between the lines; drop the striker to drag a centre-back out.", f"gap between lines {m['interline']:.0f} m")
        rule("COMPACT", m["compactness"] >= t.compact_hi, 2, "Very compact block: avoid central entries; circulate wide to disorganise it and bait their press before attacking the vacated side.", f"compactness {m['compactness']:.2f}")
        for lane, i in ((1, 0), (3, 1)):
            side = s._side(lane)
            rule(f"HS_FREE_{side}", m["opp_hs"][i] == 0 and m["our_hs"][i] == 0, 2, f"{side.capitalize()} half-space is empty: occupy it with an interior/winger to receive on the half-turn (superioridad posicional).", f"opp in lane {m['opp_hs'][i]}, ours {m['our_hs'][i]}")
            rule(f"HS_CONGESTED_{side}", m["opp_hs"][i] >= t.hs_overload, 3, f"{side.capitalize()} half-space is congested ({m['opp_hs'][i]} opponents): do not force it - switch to the other side.", f"{m['opp_hs'][i]} opponents in lane")
        rule("PRESS_TRIGGER", m["press_n"] >= t.press_min, 1, f"Press wave on #{m['press_target']} ({m['press_n']} opponents closing): release with a third-man pass to the free man behind the first line; use the goalkeeper as an extra man or go long to an isolated forward.", f"{m['press_n']} opponents closing > {t.press_close_ms} m/s")
        rule("NO_PRESS", m["press_n"] == 0 and m["line_height"] < 35 and m["line_shift"] < 2, 3, "Opponent is not pressing: push our line and full-backs up, dominate by circulation and attract before releasing.", f"line shift {m['line_shift']:+.1f} m")
        rule("LINE_RISING", m["line_shift"] >= 4, 2, "Their line is stepping up: expect an offside trap/press - play quick one-touch combinations and be ready for the ball over the top.", f"line moved {m['line_shift']:+.1f} m over {m['window_s']:.0f}s")
        for pid, p in (profiles or {}).items():                                          # counter-measures targeted at players with NO prior stats
            if p.confidence < 0.4 or p.team < 0: continue
            if p.role in ("CB", "FB") and p.speed_index >= 0.8: R[f"FAST_DEF_{pid}"] = Recommendation(2, f"FAST_DEF_{pid}", f"#{pid} ({p.role}) is quick ({p.top_speed_ms:.1f} m/s): avoid straight races in behind against him; play to feet and combine.", f"speed_index {p.speed_index:.2f}")
            if p.role in ("CB", "FB") and p.speed_index <= 0.4 and m["line_height"] >= 30: R[f"SLOW_DEF_{pid}"] = Recommendation(2, f"SLOW_DEF_{pid}", f"#{pid} ({p.role}) is slow for a high line: target his channel with runs in behind.", f"speed_index {p.speed_index:.2f}")
            if p.role in ("CM", "WM") and p.press_resistance <= 0.35: R[f"PRESS_ME_{pid}"] = Recommendation(2, f"PRESS_ME_{pid}", f"#{pid} loses mobility under pressure: trigger our press when the ball goes to him.", f"press_resistance {p.press_resistance:.2f}")
        return m, s._hysteresis(R)
    def _hysteresis(s, R):
        """A rule must hold for `hold` consecutive evaluations to appear and be absent `release` times to disappear."""
        out = []
        for tag in set(R) | set(s.streak):
            on = tag in R; c = s.streak.get(tag, 0); c = max(c, 0) + 1 if on else min(c, 0) - 1; s.streak[tag] = c
            if on and c >= s.th.hold: s.active[tag] = R[tag]
            if not on and -c >= s.th.release: s.active.pop(tag, None); s.streak.pop(tag, None)
            elif on and tag in s.active: s.active[tag] = R[tag]
        return sorted(s.active.values(), key=lambda r: (r.priority, r.tag))
