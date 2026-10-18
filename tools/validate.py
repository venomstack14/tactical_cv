"""Validate the rule engine against ANALYST labels.   python -m tools.validate report.json labels.json
labels.json: [{"start": 12.0, "end": 31.5, "tag": "HIGH_LINE"}, ...]   (tags as in engine/tactics.py, e.g. HIGH_LINE, LOW_BLOCK, NARROW, PRESS_TRIGGER)
Compares per-0.5s agreement (precision / recall / F1 per tag). Tags with F1 < 0.7 need their Thresholds retuned. Needs real, labelled footage - this tool cannot produce that."""
import json, sys, collections
def evaluate(timeline, labels, step=0.5):
    if not timeline: return {}
    T0, T1 = timeline[0]["t"], timeline[-1]["t"]; tags = sorted({l["tag"] for l in labels} | {g for e in timeline for g in e["tags"]}); res = {}
    pred = {g: [] for g in tags}; truth = {g: [] for g in tags}; i = 0
    for k in range(int((T1 - T0) / step) + 1):
        t = T0 + k * step
        while i + 1 < len(timeline) and timeline[i + 1]["t"] <= t: i += 1                   # last engine state at or before t
        for g in tags: pred[g].append(g in timeline[i]["tags"]); truth[g].append(any(l["tag"] == g and l["start"] <= t <= l["end"] for l in labels))
    for g in tags:
        tp = sum(p and q for p, q in zip(pred[g], truth[g])); fp = sum(p and not q for p, q in zip(pred[g], truth[g])); fn = sum(q and not p for p, q in zip(pred[g], truth[g]))
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1); res[g] = {"precision": round(pr, 2), "recall": round(rc, 2), "f1": round(2 * pr * rc / max(pr + rc, 1e-9), 2), "labelled_s": round(sum(truth[g]) * step, 1), "needs_tuning": (2 * pr * rc / max(pr + rc, 1e-9)) < 0.7}
    return res
if __name__ == "__main__":
    rep, lab = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    for g, r in evaluate(rep["timeline"], lab).items(): print(f"{g:16s} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f} ({r['labelled_s']}s labelled){'  <- retune thresholds' if r['needs_tuning'] else ''}")
