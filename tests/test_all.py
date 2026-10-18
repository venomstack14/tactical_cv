"""Self-contained checks on a synthetic broadcast clip with known truth.  Run:  python -m tests.test_all   (or pytest)"""
import numpy as np, cv2
from core import pitch
from tools import make_demo_video as G
from vision.calibration import PitchCalibrator, solve_from_keypoints
from vision.tracker import decode_yolov8, ByteTrack, PlayerProfiler
from engine.tactics import TacticsEngine, Snapshot

def test_template_in_bounds():
    for p in pitch.polylines(): assert p[:, 0].min() >= -1e-6 and p[:, 0].max() <= pitch.L + 1e-6 and p[:, 1].min() >= -1e-6 and p[:, 1].max() <= pitch.W + 1e-6
def test_keypoint_homography_rejects_bad_input():
    assert solve_from_keypoints([[0, 0], [1, 1]], [[0, 0], [1, 1]])[0] is None                     # too few points
    H = G.H_pitch2img(*G.camera(0, 8)); proj = lambda ns: cv2.perspectiveTransform(np.float32([[pitch.LANDMARKS[n] for n in ns]]), H)[0]
    collinear = ["center_spot", "circle_near", "circle_far", "halfway_near", "halfway_far"]          # all on the halfway line -> degenerate
    assert solve_from_keypoints(proj(collinear), [pitch.LANDMARKS[n] for n in collinear])[1] != "ok"
    good = collinear + ["left_box_far", "left_box_near", "right_box_near"]; Hs, why = solve_from_keypoints(proj(good), [pitch.LANDMARKS[n] for n in good]); assert why == "ok"
def test_calibration_tracks_camera_pan():
    names = ["center_spot", "circle_near", "circle_far", "halfway_near", "halfway_far", "left_box_far", "left_box_near", "right_box_near"]; img, H0, _ = G.render(0, 8)
    corr = [(cv2.perspectiveTransform(np.float32([[pitch.LANDMARKS[n]]]), H0)[0, 0], pitch.LANDMARKS[n]) for n in names]; corr = [c for c in corr if 5 < c[0][0] < 955 and 5 < c[0][1] < 535]
    cal = PitchCalibrator((960, 540)); cal.init_from_correspondences(corr); errs = []
    for i in range(0, 100):
        im, Ht, tr = G.render(i / 25, 8); st = cal.update(im, [p["box"] for p in tr.values()])
        g = np.float32([[x, y] for x in range(20, 90, 10) for y in range(10, 60, 10)]); px = cv2.perspectiveTransform(g.reshape(-1, 1, 2), Ht).reshape(-1, 2); v = (px[:, 0] > 20) & (px[:, 0] < 940) & (px[:, 1] > 250)
        errs.append(np.linalg.norm(cv2.perspectiveTransform(px[v].reshape(-1, 1, 2), st.H).reshape(-1, 2) - g[v], axis=1).mean())
    assert np.mean(errs) < 0.5, np.mean(errs)
def test_yolo_decode():
    o = np.zeros((1, 84, 100), np.float32); o[0, :4, 3] = [200, 200, 40, 90]; o[0, 4, 3] = .9; d = decode_yolov8(o, 2.0, 10, 0); assert len(d.persons) == 1 and abs(d.persons[0, 0] - 85) < 1
def test_bytetrack_keeps_ids():
    bt = ByteTrack(); ids = set()
    for k in range(30):
        dets = np.array([[100 + 3 * k, 100, 130 + 3 * k, 190, .9], [400 - 2 * k, 120, 430 - 2 * k, 210, .9]], float)
        for t in bt.update(dets if k % 2 == 0 else None): ids.add(t.id)
    assert ids == {1, 2}
def test_profiler_speed_and_role():
    pf = PlayerProfiler()
    for k in range(120): pf.update(k * .1, 1, (20 + 5 * k * .1, 34), rel=np.array([-15.0, 0.0]))   # 5 m/s centre-back-like
    p = pf.profile(1); assert abs(p.top_speed_ms - 5.0) < .3 and p.role == "CB"
def _scene(h, ws=1.0):
    f = lambda ys: [34 + (y - 34) * ws for y in ys]; o = {0: (104, 34)}
    for i, y in enumerate(f((10, 25, 43, 58))): o[1 + i] = (105 - h, y)
    for i, y in enumerate(f((16, 34, 52))): o[5 + i] = (105 - h - 14, y)
    for i, y in enumerate(f((20, 34, 48))): o[8 + i] = (105 - h - 26, y)
    return o
def _tags(h, ws=1.0):
    e = TacticsEngine(+1)
    for k in range(60):
        e.push(Snapshot(k * .1, {1: (40, 30)}, _scene(h, ws)))
        if k % 5 == 4: m, r = e.evaluate()
    return m, {x.tag for x in r}
def test_rules():
    assert "HIGH_LINE" in _tags(45)[1] and "LOW_BLOCK" in _tags(20)[1] and "NARROW" in _tags(33, .55)[1]
    m, _ = _tags(45); assert abs(m["line_height"] - 45) < 0.5
def test_low_coverage_is_suppressed():
    e = TacticsEngine(+1)
    for k in range(30): e.push(Snapshot(k * .1, {}, {i: (80, 30 + i) for i in range(3)}))
    m, r = e.evaluate(); assert m is None and r[0].tag == "LOW_COVERAGE"
# ---- v2 additions: match stats, jersey voting, validation harness
def test_matchstats_and_identity_and_validate():
    import numpy as np
    from analytics.matchstats import MatchStats
    from vision.identity import JerseyReader, TeamSheet
    from tools.validate import evaluate
    ms = MatchStats()                                   # team 0: P1 -> P2 (10 m pass), then team 1 (P3) takes the ball
    pl = {1: ((10, 30), 0), 2: ((20, 30), 0), 3: ((22, 30), 1), 9: ((80, 10), 1)}
    for k in range(150):
        t_ = k * 0.1; ball = [(10, 30), (20, 30), (22, 30)][min(int(t_ // 1.5), 2)]
        ms.update(t_, ball if k % 2 == 0 else None, pl)
    s = ms.summary(0); assert s["passes"]["ours"] >= 1, s; assert s["possession_pct"]["ours"] > 0 and s["turnovers"]["ours"] == 1, s
    jr = JerseyReader(min_h=10, every=1, ocr=lambda c: 7)   # stubbed OCR: voting + sheet naming
    fr = np.zeros((200, 100, 3), np.uint8)
    for i in range(6): jr.observe(fr, 5, (0, 0, 100, 200), i)
    ids = jr.identities({5: 0}, 0, None); assert ids[5]["number"] == 7
    jr2 = JerseyReader(min_h=10, every=1, ocr=lambda c: None); jr2.observe(fr, 6, (0, 0, 100, 200), 0); assert jr2.number(6) is None
    tl = [{"t": i * .5, "tags": ["HIGH_LINE"] if i < 20 else []} for i in range(40)]
    r = evaluate(tl, [{"start": 0, "end": 9.5, "tag": "HIGH_LINE"}]); assert r["HIGH_LINE"]["f1"] > 0.95, r

def test_tendency_and_forecast():
    from analytics.tendency import analyse
    fr = []                                              # ours attacks +x. P1 keeps surging forward, P2 stays deep, 4 mid players filler; opp sits then pushes
    for k in range(300):
        t = k * 0.2; s = (k // 15) % 2; run = 40 + (t % 3) * 10 * (1 if s == 0 else -1) + 20
        ours = [(1, 0, 60 + 8 * np.sin(t)), (2, 0, 20), (3, 0, 45), (4, 0, 48), (5, 0, 50), (6, 0, 52), (7, 0, 44)]
        ours[0] = (1, 0, 55 + (t % 4) * 4)
        pushed = 100 - (65 if (k // 50) % 2 else 35)
        opp = [(10 + i, 1, pushed + i * 2) for i in range(7)]
        fr.append({"t": round(t, 1), "p": [[i, tm, x, 20 + i * 4] for i, tm, x in ours] + [[i, tm, x, 20 + i * 4] for i, tm, x in opp], "b": None})
    r = analyse(fr, 0, +1); d = {p["id"]: p for p in r["players"]}
    assert d[1]["attack_score"] > d[2]["attack_score"] and d[2]["label"] == "Defensive", d
    assert r["phases"] and r["forecast"] and set(r["forecast"][-1]["next"]) <= set(__import__("analytics.tendency", fromlist=["PH"]).PH)

def test_nim_client_with_mock_server():
    import threading, json as J, http.server
    from vision.nim import NimClient
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(s, *a): pass
        def do_POST(s):
            b = J.loads(s.rfile.read(int(s.headers["Content-Length"]))); assert s.headers["Authorization"] == "Bearer k"; txt = str(b["messages"])
            out = '{"number": 7, "legible": true}' if "NUMBER" in txt else '{"role": "referee"}' if "Classify" in txt else "Brief."
            s.send_response(200); s.send_header("Content-Type", "application/json"); s.end_headers(); s.wfile.write(J.dumps({"choices": [{"message": {"content": out}}]}).encode())
    srv = http.server.HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = NimClient(key="k", base="http://127.0.0.1:%d" % srv.server_port, budget=3); img = np.full((120, 60, 3), 90, np.uint8)
    assert c.jersey_ocr(img) == 7 and c.classify(img) == "referee" and c.brief({"a": 1}) == "Brief."
    assert c.jersey_ocr(img) == 7 and c.calls == 3                                             # cached repeat costs nothing
    assert c.classify(np.full((100, 50, 3), 10, np.uint8)) is None and c.calls == 3             # budget exhausted -> graceful None
    assert NimClient(key="").jersey_ocr(img) is None                                             # no key -> offline
    srv.shutdown()

def test_brief_verifier_and_pressing():
    from vision.nim import verify_brief; from analytics.pressing import press_index, key_moments
    f = {"metrics": {"line_height": 36.3, "width": 47.8}}
    assert verify_brief("Line at 36 m, width 48 m.", f)["ok"] and not verify_brief("Line at 52 m.", f)["ok"]
    fr = [{"p": [[1, 0, 10, 10], [2, 0, 12, 10], [3, 0, 40, 40], [4, 1, 11, 11], [5, 1, 50, 50], [6, 1, 60, 60]]}] * 3
    assert press_index(fr, 0)["press_index"] == 1.0 and press_index(fr, 1)["press_index"] == 0.0
    assert key_moments([{"t": 0, "tags": ["LOW_COVERAGE"], "r": []}, {"t": 5, "tags": ["HIGH_LINE"], "r": [[2, "HIGH_LINE", "msg", "ev"]]}])[0]["t"] == 5

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
