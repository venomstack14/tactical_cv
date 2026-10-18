"""Real, edited broadcast footage (cuts, replays, close-ups, panning camera): shot-segment, detect+track players/ball in wide shots, classify teams.
Image-space only: with a moving/zooming camera and no pitch landmarks we do NOT claim metres, speeds or formations.
python -m tools.analyze_raw clip.mp4 out.mp4 stats.json --weights yolov8m.pt [--imgsz 1280]"""
import sys, json, argparse, time, os, pickle, cv2, numpy as np
from collections import defaultdict
from vision.reid import Tracklet, descriptor, stitch
from vision.tracker import ByteTrack, TeamClassifier, vote_team, UltralyticsDetector

def green_frac(f):
    h = cv2.cvtColor(cv2.resize(f, (160, 80)), cv2.COLOR_BGR2HSV); return float(((h[..., 0] > 30) & (h[..., 0] < 90) & (h[..., 1] > 50) & (h[..., 2] > 40)).mean())
def cam_motion(p, c, k=3):
    """Global similarity transform prev->cur (full-res px) from background features (RANSAC discards moving players). None if unreliable."""
    sm = lambda f: cv2.cvtColor(cv2.resize(f, (480, int(f.shape[0] * 480 / f.shape[1]))), cv2.COLOR_BGR2GRAY); a, b = sm(p), sm(c); sc = p.shape[1] / 480
    p0 = cv2.goodFeaturesToTrack(a, 250, 0.01, 7)
    if p0 is None or len(p0) < 12: return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None); ok = st.ravel() == 1
    if ok.sum() < 12: return None
    M, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if M is None or inl.sum() < 10: return None
    M = M.astype(np.float64); M[:, 2] *= sc; return M
def warp_tracks(bt, M):
    A = M[:, :2]; sc = float(np.sqrt(abs(np.linalg.det(A))))
    for t in bt.tracks: t.mean[:2] = A @ t.mean[:2] + M[:, 2]; t.mean[3] *= sc; t.mean[4:6] = A @ t.mean[4:6]
def sig(f): return cv2.normalize(cv2.calcHist([cv2.cvtColor(cv2.resize(f, (160, 80)), cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256]), None).flatten()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("stats"); ap.add_argument("--weights", default="yolov8m.pt"); ap.add_argument("--imgsz", type=int, default=1280); ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--cut", type=float, default=0.25); ap.add_argument("--stride", type=int, default=3, help="run detector every N frames (tracker predicts in between)"); a = ap.parse_args()
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30; frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    N, (H, W) = len(frames), frames[0].shape[:2]; det = UltralyticsDetector(a.weights, a.imgsz, a.conf)
    # ---- pass 1: cuts + detections on grass-dominated frames
    shot, shots, dets, cols, prev, g = 0, [], {}, {}, None, []
    for i, f in enumerate(frames):
        s = sig(f)
        if prev is not None and cv2.compareHist(prev, s, cv2.HISTCMP_BHATTACHARYYA) > a.cut: shot += 1
        prev = s; g.append(green_frac(f)); shots.append(shot)
    t0 = time.time(); cache = a.video + f".{a.stride}.{a.imgsz}.dets.pkl"
    if os.path.exists(cache): dets, cols = pickle.load(open(cache, "rb"))
    else:
        for i, f in enumerate(frames):
            if g[i] < 0.30 or i % a.stride: continue
            d = det.detect(f); dets[i] = d; cols[i] = [TeamClassifier.color(f, b) for b in d.persons[:, :4]]
        pickle.dump((dets, cols), open(cache, "wb"))
    print(f"detect pass {time.time() - t0:.0f}s on {len(dets)}/{N} frames")
    # ---- shot typing: wide = many people, all small
    info = {}
    for sid in sorted(set(shots)):
        idx = [i for i in range(N) if shots[i] == sid]; hs = [float(np.median((dets[i].persons[:, 3] - dets[i].persons[:, 1]) / H)) for i in idx if i in dets and len(dets[i].persons)]; ns = [len(dets[i].persons) for i in idx if i in dets]
        wide = len(hs) > 0.6 * len([i for i in idx if i in dets or g[i] >= 0.30 and i % a.stride == 0]) and np.median(hs) < 0.32 and np.median(ns) >= 3 and np.mean([g[i] for i in idx]) > 0.35
        info[sid] = {"shot": sid, "start_s": round(idx[0] / fps, 2), "end_s": round((idx[-1] + 1) / fps, 2), "type": "wide" if wide else ("close-up/other" if len(hs) else "non-pitch (crowd/graphic)"), "median_person_h": round(float(np.median(hs)), 3) if hs else None, "median_players": float(np.median(ns)) if ns else 0}
    # ---- teams: fit on every torso colour from wide shots
    tc = TeamClassifier();  tc.samples = [c for i in dets if info[shots[i]]["type"] == "wide" for c in cols[i] if c is not None and c[2] < 145]
    if len(tc.samples) >= 20: tc._fit(); tc.reject = 60.0     # generous: nearest kit centre wins; yellow officials are far (LAB) and stay unassigned
    print("wide shots:", [(s["start_s"], s["end_s"]) for s in info.values() if s["type"] == "wide"], "| team colours:", {k: tc.bgr(k) for k in (0, 1)} if tc.centers is not None else None)
    # ---- pass 2a: track (no drawing) + camera-compensated reference positions + appearance descriptors
    bt = ByteTrack(); cur = -1; ball_seen = defaultdict(int); official = {}; wide_frames = defaultdict(int); live = []; Tm = np.eye(3); tl = {}; sid_tr = defaultdict(dict)
    for i, f in enumerate(frames):
        sid = shots[i]
        if sid != cur: cur, bt, Tm = sid, ByteTrack(), np.eye(3)
        wide = info[sid]["type"] == "wide"; row = {"i": i, "t": round(i / fps, 2), "shot": sid, "type": info[sid]["type"], "n": [0, 0, 0], "ball": None, "tracks": []}
        if wide:
            if i > 0 and shots[i - 1] == sid:
                M = cam_motion(frames[i - 1], f)
                if M is not None: warp_tracks(bt, M); M3 = np.vstack([M, [0, 0, 1]]); Tm = Tm @ np.linalg.inv(M3)   # cur-frame px -> shot reference px
            sc = float(np.sqrt(abs(np.linalg.det(Tm[:2, :2])))); wide_frames[sid] += 1; d = dets.get(i); tracks = bt.update(d.persons if d is not None else None)
            for tr in tracks:
                x1, y1, x2, y2 = [int(v) for v in tr.box]; col = TeamClassifier.color(f, tr.box)
                yel = official.setdefault((sid, tr.id), [0, 0]); yel[1] += 1; yel[0] += int(col is not None and col[2] >= 138)      # strong-yellow torso (LAB b) in >=30% of a track's frames = match official
                vote_team(tr, tc.classify(col) if tc.centers is not None else -1); tm = -1 if (yel[1] >= 3 and yel[0] / yel[1] >= 0.3) else tr.team
                key = (sid, tr.id); tk = sid_tr[sid].setdefault(tr.id, Tracklet(key, tm)); tk.team = tm if tm >= 0 else tk.team
                foot = Tm @ np.array([(x1 + x2) / 2, y2, 1.0]); tk.add(i, (float(foot[0]), float(foot[1])), (y2 - y1) * sc, descriptor(f, tr.box) if i % 3 == 0 else None)
                row["n"][tm if tm >= 0 else 2] += 1; row["tracks"].append([tr.id, int(tm), x1, y1, x2, y2])
            if d is not None and d.ball: ball_seen[sid] += 1; row["ball"] = [round(d.ball[0], 1), round(d.ball[1], 1)]
        live.append(row)
    # ---- re-ID: stitch tracklets within each shot
    tk_all = {sid: [t for t in v.values() if len(t.i) >= 4] for sid, v in sid_tr.items()}; remap = {}
    for sid, ts in tk_all.items():
        if ts: m = stitch(ts, fps); remap.update({k: (k[0], m[k][1]) for k in m})
    pickle.dump((tk_all, fps), open(a.video + ".tracklets.pkl", "wb"))
    n_before = sum(len(v) for v in tk_all.values()); n_after = len(set(remap.values()))
    print(f"re-ID: {n_before} tracklets -> {n_after} identities")
    for r in live:
        for t in r["tracks"]: t[0] = remap.get((r["shot"], t[0]), (r["shot"], t[0]))[1]
        r["tracks"] = [t for t in r["tracks"] if (r["shot"], t[0]) in set(remap.values()) or True]
    # ---- pass 2b: render from the stored (re-identified) tracks
    W_ = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)); cap2 = cv2.VideoCapture(a.video)
    TXT = lambda im, s, o, sc=0.6, c=(255, 255, 255), th=1: (cv2.putText(im, s, o, 0, sc, (0, 0, 0), th + 3, cv2.LINE_AA), cv2.putText(im, s, o, 0, sc, c, th, cv2.LINE_AA))
    per_track = defaultdict(lambda: {"frames": 0, "team": defaultdict(int), "first_s": None, "last_s": None})
    for i, f in enumerate(frames):
        r = live[i]; out = f.copy(); sid = r["shot"]
        for tid, tm, x1, y1, x2, y2 in r["tracks"]:
            c = ((60, 220, 255) if tc.bgr(tm)[0] > 140 else (255, 120, 40)) if tm >= 0 else (150, 150, 150); cv2.rectangle(out, (x1, y1), (x2, y2), c, 2); TXT(out, f"#{tid}", (x1, max(y1 - 5, 12)), 0.5, (255, 255, 255))
            q = per_track[(sid, tid)]; q["frames"] += 1; q["first_s"] = q["first_s"] if q["first_s"] is not None else r["t"]; q["last_s"] = r["t"]
            if tm >= 0: q["team"][tm] += 1
        if r["ball"]: cv2.circle(out, (int(r["ball"][0]), int(r["ball"][1])), 9, (0, 255, 255), 2, cv2.LINE_AA)
        TXT(out, f"shot {sid} | {info[sid]['type']}" + (f" | players {sum(r['n'])}" if r["type"] == "wide" else ""), (10, H - 12), 0.55, (0, 255, 255)); W_.write(out)
    W_.release()
    for (sid, tid), v in per_track.items(): v["shot"] = sid
    # ---- stats
    trk = []
    for (sid, tid), v in per_track.items():
        if v["frames"] >= 8: trk.append({"shot": sid, "id": tid, "frames": v["frames"], "seconds": round(v["frames"] / fps, 2), "team": (max(v["team"], key=v["team"].get) if v["team"] else -1), "first_s": v["first_s"], "last_s": v["last_s"]})
    per_shot = []
    for sid, s in info.items():
        if s["type"] != "wide": per_shot.append(s); continue
        n = [r["n"][0] + r["n"][1] + r["n"][2] for r in live if r["shot"] == sid]; mx = int(max(n)); ids = [t for t in trk if t["shot"] == sid]
        per_shot.append(s | {"max_simultaneous_players": mx, "unique_track_ids": len(ids), "id_fragmentation": round(len(ids) / max(mx, 1), 2), "ball_visible_pct": round(100 * ball_seen[sid] * a.stride / max(wide_frames[sid], 1), 1), "mean_track_seconds": round(float(np.mean([t["seconds"] for t in ids])), 2) if ids else 0})
    json.dump({"video": a.video, "fps": fps, "size": [W, H], "duration_s": round(N / fps, 2), "weights": a.weights, "team_colors_bgr": {str(k): tc.bgr(k) for k in (0, 1)} if tc.centers is not None else None, "wide_seconds": round(sum(wide_frames.values()) / fps, 2), "shots": per_shot, "tracks": trk, "frames": live, "reid": {"tracklets": n_before, "identities": n_after}, "scope": "image-space tracking in wide shots only; no metric calibration (moving broadcast camera, edited footage)"}, open(a.stats, "w"), default=float)
    print(json.dumps({"wide_seconds": round(sum(wide_frames.values()) / fps, 1), "shots": [(s["start_s"], s["type"]) for s in per_shot]}))
main()
