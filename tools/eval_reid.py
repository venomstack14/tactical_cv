"""Measure the re-ID linker on REAL tracklets by synthetic breaks: split long tracklets with a 0.2-1 s gap, then check the linker rejoins the right halves and does not link wrongly.
Assumes the original long tracklets are mostly one player (they are ByteTrack outputs). python -m tools.eval_reid clip.mp4.tracklets.pkl"""
import sys, pickle, copy, random, numpy as np
from vision.reid import Tracklet, stitch
import json, os
KW = json.loads(os.environ.get("KW", "{}"))
tk_all, fps = pickle.load(open(sys.argv[1], "rb")); good = bad = missed = 0; trials = 0
def piece(t, a, b, key):
    n = Tracklet(key, t.team); [n.add(t.i[k], t.xy[k], t.hr[k], t.desc[min(k // 3, len(t.desc) - 1)] if t.desc else None) for k in range(a, b)]; return n
for seed in range(30):
    rnd = random.Random(seed)
    for sid, ts in tk_all.items():
        long_ = [t for t in ts if len(t.i) >= 45 and t.team >= 0]
        if len(long_) < 3: continue
        victim = rnd.choice(long_); cut = rnd.randint(15, len(victim.i) - 15); g = rnd.randint(6, 30); a = max(cut - g // 2, 8); b = min(cut + g // 2, len(victim.i) - 8)
        A, B = piece(victim, 0, a, (sid, "A")), piece(victim, b, len(victim.i), (sid, "B")); rest = [t for t in ts if t is not victim]
        m = stitch(rest + [A, B], fps, **KW); trials += 1
        if m[B.key] == m[A.key]: good += 1                                                    # correctly rejoined
        elif m[B.key] == B.key and m[A.key] == A.key: missed += 1                           # left apart (safe failure)
        else: bad += 1                                                                      # linked to a different tracklet (identity error)
print({"trials": trials, "rejoined_correctly": round(good / trials, 3), "left_apart_safe": round(missed / trials, 3), "wrong_link": round(bad / trials, 3)})
