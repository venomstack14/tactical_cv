"""Offline tracklet stitching (appearance + camera-compensated motion) to repair ID switches inside one continuous shot.
Honest scope: kits of one team look alike, so appearance only separates players weakly (skin, hair, boots, shorts/socks details). The main signal is
position continuity AFTER undoing camera pan/zoom. Links are one-to-one, never overlap in time, never cross teams, and are gated by a physical speed limit."""
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment

def descriptor(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]; h, w = y2 - y1, x2 - x1
    if h < 24 or w < 8: return None
    out = []
    for a, b in ((.18, .5), (.5, .82)):
        c = frame[max(y1 + int(a * h), 0):y1 + int(b * h), max(x1 + int(.25 * w), 0):x2 - int(.25 * w)]
        if c.size == 0: return None
        hist = cv2.calcHist([cv2.cvtColor(c, cv2.COLOR_BGR2HSV)], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel(); out.append(hist / max(hist.sum(), 1e-9))
    return np.concatenate(out).astype(np.float32)
def app_dist(d1, d2):
    bc = lambda p, q: float(np.sqrt(max(1 - np.sqrt(p * q).sum(), 0)))
    return 0.5 * (bc(d1[:48], d2[:48]) + bc(d1[48:], d2[48:]))

class Tracklet:
    """Observations are given in the shot's reference frame: ref (x,y) = foot point, hr = box height, desc, frame index."""
    def __init__(s, key, team): s.key, s.team, s.i, s.xy, s.hr, s.desc = key, team, [], [], [], []
    def add(s, i, xy, hr, d):
        s.i.append(i); s.xy.append(xy); s.hr.append(hr)
        if d is not None: s.desc.append(d)
    @property
    def start(s): return s.i[0]
    @property
    def end(s): return s.i[-1]
    def vel(s, k=8):
        n = min(k, len(s.i))
        if n < 3: return np.zeros(2)
        dt = max(s.i[-1] - s.i[-n], 1); return (np.array(s.xy[-1]) - np.array(s.xy[-n])) / dt            # ref-px per frame
    def mean_desc(s):
        if not s.desc: return None
        d = np.mean(s.desc, 0); d[:48] /= max(d[:48].sum(), 1e-9); d[48:] /= max(d[48:].sum(), 1e-9); return d

def stitch(tracklets, fps, max_gap_s=3.0, vmax=7.0, slack_m=2.5, w_app=0.5, max_cost=0.7, app_gate=0.62):
    """-> {key: merged_key}. Tracklets of ONE shot. Cost = normalised motion error + w_app * appearance distance."""
    T = sorted(tracklets, key=lambda t: t.start); A = T; B = T; C = np.full((len(A), len(B)), 1e6)
    for ia, a in enumerate(A):
        for ib, b in enumerate(B):
            if a is b or b.start <= a.end or b.start - a.end > max_gap_s * fps or a.team != b.team or a.team < 0: continue
            gap = (b.start - a.end) / fps; mpp = 1.8 / max(np.median(a.hr[-5:]), 1)                       # metres per ref-px from the player's own height
            pred = np.array(a.xy[-1]) + a.vel() * (b.start - a.end) * 0.5; dm = float(np.linalg.norm(pred - np.array(b.xy[0])) * mpp); lim = vmax * gap + slack_m
            if dm > lim: continue
            da, db = a.mean_desc(), b.mean_desc(); ad = app_dist(da, db) if da is not None and db is not None else 0.45
            if ad > app_gate: continue
            C[ia, ib] = dm / lim + w_app * ad
    r, c = linear_sum_assignment(C); link = {A[i].key: B[j].key for i, j in zip(r, c) if C[i, j] <= max_cost}
    parent = {t.key: t.key for t in T}                                                                       # follow chains to their head
    for a_, b_ in link.items(): parent[b_] = a_
    def head(k):
        while parent[k] != k: k = parent[k]
        return k
    return {t.key: head(t.key) for t in T}
