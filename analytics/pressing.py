"""Pressing index + key moments from the 5 Hz position log (proxy: no ball/event data needed).
press_index(team) = share of observed time in which >=2 of that team's players are within R m of the same opponent (a 'trap'), plus mean nearest-pressers count.
NOT true PPDA (that needs pass/defensive-action events); it is named differently on purpose."""
import numpy as np
def press_index(frames, team, R=5.0):
    hit = n = 0; cnt = []
    for f in frames:
        a = np.array([[x, y] for _, tm, x, y in f["p"] if tm == team]); b = np.array([[x, y] for _, tm, x, y in f["p"] if tm >= 0 and tm != team])
        if len(a) < 3 or len(b) < 3: continue
        c = (np.linalg.norm(b[:, None] - a[None], axis=2) < R).sum(1); n += 1; hit += int((c >= 2).any()); cnt.append(float(c.max()))
    return {"press_index": round(hit / n, 2) if n else None, "max_pressers_mean": round(float(np.mean(cnt)), 2) if cnt else None, "samples": n}
def key_moments(timeline, min_gap=2.0):
    """Moments where the rule-engine tag set changes to something non-trivial -> [{t, tags, why}] (feeds clip cutting)."""
    out, last, prev = [], -1e9, None
    for e in timeline:
        tg = [t for t in e["tags"] if t != "LOW_COVERAGE"]
        if tg and tg != prev and e["t"] - last >= min_gap: out.append({"t": e["t"], "tags": tg, "why": e["r"][0][2][:90] if e["r"] else ""}); last = e["t"]
        prev = tg
    return out
