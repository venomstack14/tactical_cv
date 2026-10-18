#!/usr/bin/env python3
"""Offline CPU football tactical analysis.   python main.py match.mp4 --calib calib.json --attack right --out out.mp4
Left: annotated video + tactical warnings.  Right: top-down radar, opponent block metrics, counter-measures, player profile vectors."""
import argparse, json, time, cv2, numpy as np
from collections import defaultdict
from core import pitch
from vision.calibration import PitchCalibrator, load_correspondences, interactive_calibration
from vision.tracker import make_detector, ByteTrack, TeamClassifier, vote_team, PlayerProfiler
from engine.tactics import TacticsEngine, Snapshot
from vision.render import draw_video, draw_radar, draw_panel
from analytics.matchstats import MatchStats
from analytics.tendency import analyse as tendency
from vision.identity import JerseyReader, TeamSheet
from vision.nim import NimClient, verify_brief
from analytics.pressing import press_index, key_moments
NAMED = {"red": (40, 40, 220), "blue": (220, 80, 30), "white": (240, 240, 240), "black": (25, 25, 25), "yellow": (30, 220, 230), "green": (60, 180, 60), "sky": (235, 190, 90), "orange": (30, 140, 250)}
def parse_color(c):  # name or #RRGGBB -> BGR
    return NAMED[c.lower()] if c.lower() in NAMED else tuple(int(c.lstrip('#')[i:i + 2], 16) for i in (4, 2, 0))

def _js(m):
    return None if m is None else {k: (v.tolist() if hasattr(v, "tolist") else (round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v)) for k, v in m.items()}

class Pipeline:
    def __init__(s, size, fps, calib_pts, detector, attack_dir=+1, our_team=0, detect_every=2, eval_every_s=0.5, our_color=None, jersey=False, nim=None):
        s.nim, s.kind = nim if (nim and nim.key) else None, {}
        s.fps, s.det, s.every, s.our, s.dir, s.our_color = fps, detector, detect_every, our_team, attack_dir, our_color
        s.cal = PitchCalibrator(size); s.cal.init_from_correspondences(calib_pts); s.bt, s.tc, s.prof = ByteTrack(), TeamClassifier(), PlayerProfiler()
        s.eng = TacticsEngine(attack_dir); s.ms_stats, s.jr, s.team_of, s.timeline, s.ball_xy, s.log, s.last_log = MatchStats(), (JerseyReader(ocr=nim.jersey_ocr, every=10) if (jersey and nim and nim.key) else JerseyReader()) if jersey else None, {}, [], None, [], -1e9; s.eval_every, s.last_eval, s.metrics, s.recs, s.i = eval_every_s, -1e9, None, [], 0; s.ms = defaultdict(float)
    def step(s, frame):
        t = s.i / s.fps; T = time.perf_counter; a = T()
        if s.our_color is not None and s.tc.centers is not None:   # resolve 'our team' = cluster closest to the given kit colour
            lab = cv2.cvtColor(np.uint8([[s.our_color]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float); s.our = int(np.linalg.norm(s.tc.centers - lab, axis=1).argmin()); s.our_color = None
        dets = s.det.detect(frame) if s.i % s.every == 0 else None; tracks = s.bt.update(dets.persons if dets is not None else None); b = T()
        st = s.cal.update(frame, [tr.box for tr in tracks]); c = T()
        if dets is not None: s.ball_xy = tuple(s.cal.to_pitch([dets.ball[:2]], st.H)[0]) if dets.ball else None   # ground projection of the ball (proxy: ignores height)
        pts = []
        for tr in tracks:
            x1, y1, x2, y2 = tr.box; xy = s.cal.to_pitch([((x1 + x2) / 2, y2)], st.H)[0]
            if (s.i + tr.id) % 3 == 0 or tr.team < 0: vote_team(tr, s.tc.classify(TeamClassifier.color(frame, tr.box)))
            if s.nim is not None and tr.id not in s.kind and (tr.box[3] - tr.box[1]) >= 45 and tr.hits >= 8 and s.nim.ok:   # one VLM check per track: drop referees/staff
                x1, y1, x2, y2 = [int(v) for v in tr.box]; s.kind[tr.id] = s.nim.classify(frame[max(y1, 0):y2, max(x1, 0):x2])
            if s.kind.get(tr.id) in ("referee", "other"): continue
            if -3 <= xy[0] <= pitch.L + 3 and -3 <= xy[1] <= pitch.W + 3: pts.append((tr, xy))
        if s.jr is not None:
            for tr, _ in pts[:3]: s.jr.observe(frame, tr.id, tr.box, s.i)                  # <=3 OCR tries/frame keeps CPU bounded
        for tr, _ in pts: s.team_of[tr.id] = tr.team
        if t - s.last_log >= 0.2: s.last_log = t; s.log.append({"t": round(t, 2), "p": [[tr.id, int(tr.team), round(float(xy[0]), 1), round(float(xy[1]), 1)] for tr, xy in pts], "b": [round(float(s.ball_xy[0]), 1), round(float(s.ball_xy[1]), 1)] if s.ball_xy is not None else None})
        s.ms_stats.update(t, s.ball_xy if dets is not None else None, {tr.id: (xy, tr.team) for tr, xy in pts})
        team = defaultdict(list)
        for tr, xy in pts:
            if tr.team >= 0: team[tr.team].append((tr.id, xy))
        cen = {k: np.mean([p for _, p in v], 0) for k, v in team.items()}; prof = {}
        for k, v in team.items():
            other = [p for kk, vv in team.items() if kk != k for _, p in vv]; sgn = s.dir if k == s.our else -s.dir
            for tid, xy in v:
                rel = (xy - cen[k]) * np.array([sgn, 1.0]); nd = min((np.hypot(*(xy - o)) for o in other), default=None); s.prof.update(t, tid, xy, rel, nd)
        s.eng.push(Snapshot(t, {i: tuple(p) for i, p in team.get(s.our, [])}, {i: tuple(p) for k, v in team.items() if k != s.our for i, p in v}))
        if t - s.last_eval >= s.eval_every:
            s.last_eval = t; opp = {tid: s.prof.profile(tid, k) for k, v in team.items() if k != s.our for tid, _ in v}
            s.metrics, s.recs = s.eng.evaluate({k: v for k, v in opp.items() if v}); s.timeline.append({"t": round(t, 2), "tags": sorted({__import__("re").sub(r"_\d+$", "", r.tag) for r in s.recs}), "m": _js(s.metrics), "r": [[r.priority, r.tag, r.message, r.evidence] for r in s.recs]}); 
        d = T(); s.ms["detect+track"] += b - a; s.ms["calib"] += c - b; s.ms["analytics"] += d - c; s.i += 1
        return dict(tracks=[tr for tr, _ in pts], xy={tr.id: xy for tr, xy in pts}, calib_valid=st.valid, calib_q=st.quality, refined=st.refined, H=st.H)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video"); ap.add_argument("--calib", help="JSON of pixel<->pitch correspondences (processing-frame pixels)"); ap.add_argument("--interactive", action="store_true", help="click landmarks on frame 0")
    ap.add_argument("--detector", default="auto", choices=["auto", "blob", "onnx", "ultralytics"]); ap.add_argument("--weights", help="yolov8n.onnx (cv2.dnn) or .pt"); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--width", type=int, default=960); ap.add_argument("--detect-every", type=int, default=2, help="run the detector every N frames; Kalman fills the rest")
    ap.add_argument("--attack", default="right", choices=["right", "left"], help="direction OUR team attacks (in the camera's left->right pitch frame)"); ap.add_argument("--our-team", type=int, default=0, help="colour cluster id (0/1) that is our team"); ap.add_argument("--our-color", help="our kit colour (red|blue|white|...|#RRGGBB); overrides --our-team")
    ap.add_argument("--nim", action="store_true", help="use NVIDIA NIM (needs env NVIDIA_API_KEY): VLM jersey OCR + referee filter + AI scouting brief"); ap.add_argument("--nim-budget", type=int, default=150, help="hard cap on API calls per run"); ap.add_argument("--club", default="the analysing club"); ap.add_argument("--lang", default="English"); ap.add_argument("--jersey", action="store_true", help="try to read jersey numbers (OCR)"); ap.add_argument("--team-sheet", help="club JSON {ours:{7:name},opp:{..}} to name players by jersey number"); ap.add_argument("--out"); ap.add_argument("--report"); ap.add_argument("--max-frames", type=int, default=0); ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=True); a = ap.parse_args()
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; ok, f0 = cap.read(); assert ok, "cannot read video"
    sz = (a.width, int(round(f0.shape[0] * a.width / f0.shape[1]))); rs = lambda f: cv2.resize(f, sz, interpolation=cv2.INTER_AREA); f0 = rs(f0)
    pts = interactive_calibration(f0) if a.interactive else load_correspondences(a.calib) if a.calib else None
    if not pts: raise SystemExit("need --calib calib.json or --interactive (>=4 known pitch points)")
    nim = NimClient(budget=a.nim_budget) if a.nim else None
    if a.nim and not nim.key: print("--nim given but NVIDIA_API_KEY is not set: running fully offline")
    P = Pipeline(sz, fps, pts, make_detector(a.detector, a.weights, a.imgsz), +1 if a.attack == "right" else -1, a.our_team, a.detect_every, our_color=parse_color(a.our_color) if a.our_color else None, jersey=a.jersey or a.nim, nim=nim)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (sz[0] + 480, sz[1])) if a.out else None; cap.set(cv2.CAP_PROP_POS_FRAMES, 0); n, ema, t_prev, valid = 0, fps, time.perf_counter(), 0
    while True:
        ok, fr = cap.read()
        if not ok or (a.max_frames and n >= a.max_frames): break
        if n % 50 == 0: print(f"\r  frame {n}", end="", flush=True)
        r = P.step(rs(fr)); now = time.perf_counter(); ema = 0.9 * ema + 0.1 / max(now - t_prev, 1e-6); t_prev = now; n += 1; valid += r["calib_valid"]
        tb = lambda team: P.tc.bgr(team); opp_ids = [t for t in r["tracks"] if t.team >= 0 and t.team != P.our]
        profs = [p for p in (P.prof.profile(t.id, t.team) for t in opp_ids) if p]
        left = draw_video(rs(fr), r["tracks"], r, tb, P.recs, ema, P.our); radar = draw_radar((480, 310), [(t.id, t.team, *r["xy"][t.id]) for t in r["tracks"]], None, tb, P.our)
        view = np.hstack([left, draw_panel(radar, P.metrics, P.recs, profs, sz[1])])
        if writer: writer.write(view)
        if a.show:
            try:
                cv2.imshow("tactical analysis (q to quit)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            except cv2.error: print("no display available - continuing headless"); a.show = False
    el = sum(P.ms.values()); print(f"{n} frames | {n / max(el, 1e-9):.1f} FPS pipeline-only (detect+track {1000 * P.ms['detect+track'] / n:.1f} ms, calib {1000 * P.ms['calib'] / n:.1f} ms, analytics {1000 * P.ms['analytics'] / n:.1f} ms) | calibration valid {valid}/{n}")
    if P.tc.centers is not None: print("team colours (BGR):", {k: P.tc.bgr(k) for k in (0, 1)}, "| our team cluster =", P.our)
    if P.metrics: print("opponent block:", {k: (v.tolist() if hasattr(v, "tolist") else round(v, 2)) for k, v in P.metrics.items()})
    for r in P.recs: print(r)
    if a.report:
        profs = [p.__dict__ | {"vector": p.vector().tolist()} for p in (P.prof.profile(i, P.team_of.get(i, -1)) for i in P.prof.p) if p]
        ids = P.jr.identities(P.team_of, P.our, TeamSheet(a.team_sheet) if a.team_sheet else None) if P.jr else {}
        for p in profs: p["identity"] = ids.get(p["id"])
        tt = {}                                                                         # team physical totals (only tracked, visible players; sum of per-track distance)
        for k, nm in ((P.our, "ours"), (1 - P.our, "opp")):
            q = [p for p in profs if p["team"] == k and p["observed_s"] >= 5]
            tt[nm] = {"players_observed": len(q), "distance_m": round(sum(p["distance_m"] for p in q), 1), "top_speed_kmh": round(3.6 * max([p["top_speed_ms"] for p in q], default=0), 1), "sprints": sum(p["sprints"] for p in q)}
        json.dump({"metrics": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in (P.metrics or {}).items()}, "recommendations": [r.__dict__ for r in P.recs], "profiles": profs, "team_stats": tt, "match_stats": P.ms_stats.summary(P.our), "timeline": P.timeline, "frames": P.log, "our": P.our, "attack": P.dir, "tendency": tendency(P.log, P.our, P.dir), "fps": fps, "duration": round(n / fps, 1), "team_colors": {k: "#%02x%02x%02x" % tuple(P.tc.bgr(k)[::-1]) for k in (0, 1)}, "pressing": {"ours": press_index(P.log, P.our), "opp": press_index(P.log, 1 - P.our)}, "moments": key_moments(P.timeline), "identified_players": len(ids), "nim": ({"calls": nim.calls, "errors": nim.errors, "vlm": nim.vlm} if nim and nim.key else None), "referees_filtered": sum(1 for v in P.kind.values() if v in ("referee", "other")), "scope": "proxies from visible players only; IDs are anonymous unless jersey+team-sheet matched"}, open(a.report, "w"), indent=1, default=float)
    if writer: writer.release()
    if a.report and nim and nim.ok:
        R = json.load(open(a.report)); facts = {k: R.get(k) for k in ("metrics", "team_stats", "match_stats", "tendency", "duration")} | {"tags": R["timeline"][-6:], "recommendations": [[r["priority"], r["message"], r["evidence"]] for r in R["recommendations"]]}
        R["ai_brief"] = nim.brief(facts, a.lang, a.club); R["ai_brief_check"] = verify_brief(R["ai_brief"], facts) if R["ai_brief"] else None; json.dump(R, open(a.report, "w"), indent=1, default=float); print("AI brief:", "written to report.json" if R["ai_brief"] else "unavailable (offline fallback)", f"| NIM calls {nim.calls}/{nim.budget}")
if __name__ == "__main__": main()
