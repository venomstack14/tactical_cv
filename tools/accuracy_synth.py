"""MEASURED accuracy on the synthetic clip with exact ground truth: per-frame pitch-position error of tracked players (Hungarian match to truth) and team-label accuracy.
Synthetic only - say so when quoting it.   python -m tools.accuracy_synth report.json demo_truth.json"""
import sys, json, numpy as np
from scipy.optimize import linear_sum_assignment
R = json.load(open(sys.argv[1])); T = json.load(open(sys.argv[2])); fps = R["fps"]; err, ok, tot, rec = [], 0, 0, []
for f in R["frames"]:
    tr = T[min(int(round(f["t"] * fps)), len(T) - 1)]["players"]; tid = list(tr); gt = np.array([tr[k]["xy"] for k in tid]); gtt = [tr[k]["team"] for k in tid]; keep = [i for i, g in enumerate(gt) if 0 <= g[0] <= 105 and 0 <= g[1] <= 68]
    if not f["p"] or not keep: continue
    pr = np.array([[x, y] for _, _, x, y in f["p"]]); C = np.linalg.norm(gt[keep][:, None] - pr[None], axis=2); r, c = linear_sum_assignment(C); m = C[r, c] < 5
    err += list(C[r, c][m]); rec.append(m.sum() / len(keep))
    # team consistency: map predicted team ids onto truth letters by majority, report agreement
    pairs = [(f["p"][j][1], gtt[keep[i]]) for i, j in zip(r[m], c[m])]; tot += len(pairs)
    ok += max(sum(1 for p, g in pairs if (p == 0) == (g == "A")), sum(1 for p, g in pairs if (p == 0) != (g == "A")))
e = np.array(err); print(json.dumps({"scope": "SYNTHETIC clip, exact truth", "mean_err_m": round(e.mean(), 2), "median_err_m": round(float(np.median(e)), 2), "p90_err_m": round(float(np.percentile(e, 90)), 2), "detection_recall": round(float(np.mean(rec)), 3), "team_label_accuracy": round(ok / max(tot, 1), 3), "n_matches": int(len(e))}))
