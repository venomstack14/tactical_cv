"""Player attacking/defensive tendency, team phase timeline and a pattern-based next-phase forecast, from tracked positions only.
Everything is a heuristic over what the camera showed in THIS clip - not a trained predictor. Forecast confidence is reported."""
import numpy as np
from collections import defaultdict
L = 105.0
PH = ["DEFEND", "DEFENSIVE_TRANSITION", "BUILD_UP", "ATTACKING_TRANSITION", "ATTACK"]
def phase_of(c, v):
    """c: team centroid depth (m from own goal line, keeper excluded); v: change of c over the last 3 s (m/s)."""
    return "ATTACK" if c >= 55 else "DEFEND" if c <= 40 else "ATTACKING_TRANSITION" if v >= 1.0 else "DEFENSIVE_TRANSITION" if v <= -1.0 else "BUILD_UP"
def team_features(advS, yS, advO, c3):
    """Shared by training and inference. advS/yS: subject team players (depth, lateral); advO: other team depths (in THEIR own frame); c3: subject centroid 3 s ago or None."""
    a = np.sort(np.asarray(advS, float)); o = np.sort(np.asarray(advO, float)); a = a[1:] if len(a) >= 5 else a; o = o[1:] if len(o) >= 5 else o
    if len(a) < 3 or len(o) < 3: return None
    c, cO = float(a.mean()), float(o.mean()); v = (c - c3) / 3.0 if c3 is not None else 0.0; ph = phase_of(c, v)
    return np.array([c, v, float(np.ptp(yS)), float(np.ptp(a)), c + cO - L] + [float(ph == q) for q in PH]), ph, c
def _adv(x, sgn): return x if sgn > 0 else L - x          # metres advanced from the team's own goal line

def analyse(frames, our, our_dir, min_obs_s=5.0):
    sg = {our: our_dir, 1 - our: -our_dir}; S, T, cen = defaultdict(list), {}, {0: {}, 1: {}}
    for f in frames:
        for k in (0, 1):
            a = sorted(_adv(x, sg[k]) for _, tm, x, y in f["p"] if tm == k); a = a[1:] if len(a) >= 5 else a      # drop deepest (keeper)
            if len(a) >= 3: cen[k][f["t"]] = float(np.mean(a))
        for pid, tm, x, y in f["p"]:
            if tm >= 0: S[pid].append((f["t"], _adv(x, sg[tm]), y)); T[pid] = tm
    players = []
    for pid, s in S.items():
        obs = len(s) * 0.2
        if obs < min_obs_s: continue
        tm = T[pid]; rel = [a - cen[tm][t] for t, a, _ in s if t in cen[tm]]
        if not rel: continue
        fr = dr = 0; cool = -1e9; ts = np.array([t for t, _, _ in s]); av = np.array([a for _, a, _ in s])
        for i in range(len(s)):
            j = np.searchsorted(ts, ts[i] - 3.0)
            if j >= i or ts[i] - ts[j] < 2.4 or ts[i] < cool: continue
            d = av[i] - av[j]
            if d >= 8: fr += 1; cool = ts[i] + 3
            elif d <= -8: dr += 1; cool = ts[i] + 3
        f3, o3 = float((av >= 70).mean()), float((av <= 35).mean()); r = float(np.mean(rel))
        sc = 0.5 + 0.30 * np.clip(r / 20, -1, 1) + 0.12 * np.clip((fr - dr) / 4, -1, 1) + 0.08 * np.clip((f3 - o3) * 2, -1, 1)
        players.append({"id": pid, "team": tm, "attack_score": round(float(sc), 2), "label": "Attacking" if sc >= .62 else "Defensive" if sc <= .38 else "Balanced",
                        "rel_depth_m": round(r, 1), "fwd_runs": fr, "drop_runs": dr, "final_third_pct": round(100 * f3), "own_third_pct": round(100 * o3), "observed_s": round(obs, 1), "confidence": round(min(1.0, obs / 20), 2)})
    # ---- team phases, sampled every 1 s
    def phase(k, t):
        c = cen[k].get(t); p = cen[k].get(t - 3.0) if t >= 3 else None
        return None if c is None else phase_of(c, (c - p) / 3.0 if p is not None else 0.0)
    tgrid = sorted({round(f["t"]) for f in frames}); fmap = {round(f["t"], 1): f for f in frames}
    near = lambda t: min(cen[0].keys() | cen[1].keys(), key=lambda u: abs(u - t)) if (cen[0] or cen[1]) else t
    phases, tr = [], defaultdict(lambda: defaultdict(int)); chan = np.zeros(3); prev = None; fc = []; opp = 1 - our
    for t in tgrid:
        tt = near(t); po, pp = phase(our, tt), phase(opp, tt)
        if po is None or pp is None: continue
        phases.append({"t": t, "ours": po, "opp": pp})
        if prev is not None and pp != prev: tr[prev][pp] += 1
        prev = pp
        f = fmap.get(round(tt, 1))
        if f and pp in ("ATTACK", "ATTACKING_TRANSITION"):
            xs = sorted((_adv(x, sg[opp]), y) for _, tm, x, y in f["p"] if tm == opp)
            for a, y in xs[len(xs) // 2:]: chan[min(int(y // (68 / 3)), 2)] += 1
        row = tr[pp]; tot = sum(row.values()); n_other = len(PH) - 1
        nxt = {q: (row.get(q, 0) + .5) / (tot + .5 * n_other) for q in PH if q != pp}
        top = dict(sorted(nxt.items(), key=lambda kv: -kv[1])[:3])
        ch = None
        if chan.sum() >= 30: i = int(chan.argmax()); ch = {"name": ["near-touchline", "central", "far-touchline"][i], "share": round(float(chan[i] / chan.sum()), 2)}
        fc.append({"t": t, "cur": pp, "next": {k: round(v, 2) for k, v in top.items()}, "n": tot, "conf": "low" if tot < 3 else "medium" if tot < 8 else "higher", "channel": ch})
    return {"players": players, "phases": phases, "forecast": fc,
            "note": "heuristics on this clip only; forecast = observed phase-to-phase habits of the opponent so far, low confidence on short clips"}
