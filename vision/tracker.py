"""Detection, ByteTrack, team colour assignment and zero-shot player profiling (CPU only, no API calls).

Detectors (all return `Detections`):
  * OnnxYoloDetector  - YOLOv8n exported to ONNX, run with cv2.dnn (no torch at runtime).  Export once:  yolo export model=yolov8n.pt format=onnx imgsz=640
  * UltralyticsDetector - uses the ultralytics package if installed (optional).
  * PitchBlobDetector - classical fallback (green-pitch subtraction); needs no weights, good for clean footage / testing.
ByteTrack is implemented here (Kalman + two-stage IoU association) so there is no `supervision`/`lap` dependency."""
import cv2, numpy as np
from collections import deque
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from vision.calibration import PitchSegmenter

# ----------------------------------------------------------------------------- detection
@dataclass
class Detections:
    persons: np.ndarray                      # (N,5) x1,y1,x2,y2,score (players, referees, goalkeepers)
    ball: tuple | None = None                # (cx, cy, score) or None

def letterbox(img, size):
    h, w = img.shape[:2]; s = min(size / h, size / w); nh, nw = int(round(h * s)), int(round(w * s)); out = np.full((size, size, 3), 114, np.uint8)
    py, px = (size - nh) // 2, (size - nw) // 2; out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh)); return out, s, px, py
def decode_yolov8(out, scale, px, py, conf=0.25, iou=0.5, person=0, ball=32):
    """out: raw YOLOv8 ONNX output (1,84,N) -> Detections in original-image pixels."""
    o = out.reshape(out.shape[-2], out.shape[-1]).T                                  # (N,84): cx,cy,w,h,80 class scores
    cx, cy, w, h = o[:, 0], o[:, 1], o[:, 2], o[:, 3]; boxes = np.c_[cx - w / 2, cy - h / 2, w, h]; res = []
    for cls in (person, ball):
        sc = o[:, 4 + cls]; k = np.where(sc >= conf)[0]
        if len(k) == 0: res.append(np.zeros((0, 5))); continue
        keep = cv2.dnn.NMSBoxes(boxes[k].tolist(), sc[k].tolist(), conf, iou); keep = np.array(keep).reshape(-1).astype(int); b = boxes[k][keep]
        res.append(np.c_[(b[:, 0] - px) / scale, (b[:, 1] - py) / scale, (b[:, 0] + b[:, 2] - px) / scale, (b[:, 1] + b[:, 3] - py) / scale, sc[k][keep]])
    bl = res[1]; return Detections(res[0], None if len(bl) == 0 else (float((bl[bl[:, 4].argmax(), 0] + bl[bl[:, 4].argmax(), 2]) / 2), float((bl[bl[:, 4].argmax(), 1] + bl[bl[:, 4].argmax(), 3]) / 2), float(bl[:, 4].max())))
class OnnxYoloDetector:
    def __init__(s, path, imgsz=640, conf=0.3, iou=0.5, threads=0):
        s.net = cv2.dnn.readNetFromONNX(path); s.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV); s.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        if threads: cv2.setNumThreads(threads)
        s.sz, s.conf, s.iou = imgsz, conf, iou
    def detect(s, frame):
        img, sc, px, py = letterbox(frame, s.sz); s.net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (s.sz, s.sz), swapRB=True)); return decode_yolov8(s.net.forward(), sc, px, py, s.conf, s.iou)
class UltralyticsDetector:
    def __init__(s, weights="yolov8n.pt", imgsz=640, conf=0.3):
        from ultralytics import YOLO; s.m, s.sz, s.conf = YOLO(weights), imgsz, conf
    def detect(s, frame):
        r = s.m(frame, imgsz=s.sz, conf=s.conf, classes=[0, 32], device="cpu", verbose=False)[0]; b = r.boxes.data.cpu().numpy(); p = b[b[:, 5] == 0][:, :5]; bl = b[b[:, 5] == 32]
        return Detections(p, None if len(bl) == 0 else (float((bl[0, 0] + bl[0, 2]) / 2), float((bl[0, 1] + bl[0, 3]) / 2), float(bl[0, 4])))
class PitchBlobDetector:
    """Players = non-green blobs inside the filled pitch region after thin lines are removed by morphological opening."""
    def __init__(s, min_h=10, min_area=40): s.seg, s.min_h, s.min_area = PitchSegmenter(), min_h, min_area
    def detect(s, frame):
        green = s.seg.pitch_mask(frame); region = s.seg.fill(green, 31); fg = cv2.bitwise_and(region, cv2.bitwise_not(green))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)))
        n, _, st, _ = cv2.connectedComponentsWithStats(fg); out = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a >= s.min_area and h >= s.min_h and 1.0 <= h / max(w, 1) <= 5.5: out.append([x, y, x + w, y + h, 0.9])
        return Detections(np.array(out, float).reshape(-1, 5))
def make_detector(kind="auto", weights=None, imgsz=640):
    if kind in ("auto", "onnx") and weights and weights.endswith(".onnx"): return OnnxYoloDetector(weights, imgsz)
    if kind in ("auto", "ultralytics") and weights and weights.endswith(".pt"): return UltralyticsDetector(weights, imgsz)
    if kind in ("onnx", "ultralytics"): raise FileNotFoundError("detector weights required for kind=%s" % kind)
    return PitchBlobDetector()

# ----------------------------------------------------------------------------- ByteTrack
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1]); x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None); ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1]); return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)
class _KF:
    """Constant-velocity Kalman filter on [cx, cy, aspect, h] (as in SORT/ByteTrack)."""
    F = np.eye(8); F[:4, 4:] = np.eye(4); Hm = np.eye(4, 8); wp, wv = 1 / 20, 1 / 160
    @classmethod
    def init(c, z): h = z[3]; return np.r_[z, np.zeros(4)], np.diag(np.square([2 * c.wp * h] * 2 + [1e-2, 2 * c.wp * h] + [10 * c.wv * h] * 2 + [1e-5, 10 * c.wv * h]))
    @classmethod
    def predict(c, m, P): h = m[3]; Q = np.diag(np.square([c.wp * h] * 2 + [1e-2, c.wp * h] + [c.wv * h] * 2 + [1e-5, c.wv * h])); return c.F @ m, c.F @ P @ c.F.T + Q
    @classmethod
    def update(c, m, P, z):
        h = m[3]; R = np.diag(np.square([c.wp * h] * 2 + [1e-1, c.wp * h])); S = c.Hm @ P @ c.Hm.T + R; K = P @ c.Hm.T @ np.linalg.inv(S); return m + K @ (z - c.Hm @ m), P - K @ S @ K.T
def _z(b): w, h = b[2] - b[0], b[3] - b[1]; return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])
@dataclass
class Track:
    id: int; mean: np.ndarray; cov: np.ndarray; score: float; hits: int = 1; since: int = 0; lost: bool = False
    team: int = -1; votes: dict = field(default_factory=dict)
    @property
    def box(s): cx, cy, a, h = s.mean[:4]; w = a * h; return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
class ByteTrack:
    """update(dets) associates; update(None) only advances the Kalman prediction (use between detector calls to save CPU)."""
    def __init__(s, high=0.5, low=0.1, new=0.6, match_iou=0.2, low_iou=0.5, buffer=30, min_hits=2): s.cfg = (high, low, new, match_iou, low_iou, buffer, min_hits); s.tracks, s.next = [], 1
    def _match(s, tr, boxes, thr):
        if not tr or len(boxes) == 0: return [], list(range(len(tr))), list(range(len(boxes)))
        iou = iou_matrix(np.array([t.box for t in tr]), boxes); r, c = linear_sum_assignment(1 - iou); m = [(i, j) for i, j in zip(r, c) if iou[i, j] >= thr]
        return m, [i for i in range(len(tr)) if i not in {a for a, _ in m}], [j for j in range(len(boxes)) if j not in {b for _, b in m}]
    def _upd(s, t, d): t.mean, t.cov = _KF.update(t.mean, t.cov, _z(d[:4])); t.score, t.hits, t.since, t.lost = d[4], t.hits + 1, 0, False
    def update(s, dets):
        high, low, new, miou, liou, buf, mh = s.cfg
        for t in s.tracks: t.mean, t.cov = _KF.predict(t.mean, t.cov)
        if dets is not None:
            hi, lo = dets[dets[:, 4] >= high], dets[(dets[:, 4] >= low) & (dets[:, 4] < high)]
            m, ut, uh = s._match(s.tracks, hi[:, :4], miou)
            for i, j in m: s._upd(s.tracks[i], hi[j])
            rem = [s.tracks[i] for i in ut if not s.tracks[i].lost]                # stage 2: still-tracked, unmatched -> low-score dets
            m2, ut2, _ = s._match(rem, lo[:, :4], liou)
            for i, j in m2: s._upd(rem[i], lo[j])
            for i in ut2: rem[i].lost = True
            for j in uh:
                if hi[j, 4] >= new: z = _z(hi[j, :4]); mu, P = _KF.init(z); s.tracks.append(Track(s.next, mu, P, hi[j, 4])); s.next += 1
            for t in s.tracks:
                if t.lost: t.since += 1
            s.tracks = [t for t in s.tracks if t.since <= buf]
        return [t for t in s.tracks if not t.lost and t.hits >= mh]

# ----------------------------------------------------------------------------- team colours
class TeamClassifier:
    """Online 2-means on torso colour (LAB). Warm-up collects samples, then fixes two centres; far-off colours (referee/keeper) -> team -1."""
    def __init__(s, warmup=40): s.warmup, s.samples, s.centers, s.reject = warmup, [], None, 0.0
    @staticmethod
    def color(frame, box):
        x1, y1, x2, y2 = [int(v) for v in box]; w, h = x2 - x1, y2 - y1; c = frame[max(y1 + int(.2 * h), 0):y1 + int(.55 * h), max(x1 + int(.25 * w), 0):x2 - int(.25 * w)]
        return None if c.size == 0 else np.median(cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3), 0).astype(np.float32)
    def classify(s, col):
        if col is None: return -1
        if s.centers is None:
            s.samples.append(col)
            if len(s.samples) >= s.warmup: s._fit()
            return -1
        d = np.linalg.norm(s.centers - col, axis=1); return int(d.argmin()) if d.min() <= s.reject else -1
    def _fit(s):
        X = np.float32(s.samples); _, lab, C = cv2.kmeans(X, 2, None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 8, cv2.KMEANS_PP_CENTERS)
        s.centers = C; s.reject = 2.5 * float(np.median(np.linalg.norm(X - C[lab.ravel()], axis=1))) + 10
    def bgr(s, team):
        if s.centers is None: return (200, 200, 200)
        return tuple(int(v) for v in cv2.cvtColor(np.uint8([[s.centers[team]]]), cv2.COLOR_LAB2BGR)[0, 0])
def vote_team(track, label, min_votes=3):
    """Smooth the per-frame colour label with per-track voting (a track keeps its team unless evidence flips)."""
    if label >= 0: track.votes[label] = track.votes.get(label, 0) + 1
    if track.votes:
        best = max(track.votes, key=track.votes.get)
        if track.votes[best] >= min_votes: track.team = best
    return track.team

# ----------------------------------------------------------------------------- zero-shot player profiling
@dataclass
class PlayerProfile:
    """Dynamic 'Player Profile Vector' for a player with NO historical record - built purely from what the camera has seen."""
    id: int; team: int; role: str; role_conf: float
    speed_index: float; accel_index: float; press_resistance: float; pitch_coverage: float; work_rate: float   # all in 0..1
    top_speed_ms: float; sprints: int; distance_m: float; observed_s: float; confidence: float
    def vector(s): return np.array([s.speed_index, s.accel_index, s.press_resistance, s.pitch_coverage, s.work_rate])
SPRINT_MS, MAX_MS = 7.0, 12.0          # sprint threshold (25 km/h) and physical plausibility cap (rejects homography glitches)
CELL = 5.0                              # coverage heat-map cell size (m)
class _P:
    def __init__(s): s.h = deque(maxlen=120); s.v = deque(maxlen=40); s.acc = deque(maxlen=400); s.top = s.dist = s.obs = 0.0; s.sprints = 0; s.in_sprint = False
    s_grid = None
    def init2(s): s.grid = np.zeros((int(68 / CELL) + 1, int(105 / CELL) + 1), np.int32); s.pr_sum = s.pr_n = s.v_sum = s.v_n = 0.0; s.rel = np.zeros(2); s.relN = 0; s.dnorm = 0.0; return s
class PlayerProfiler:
    def __init__(s, window_s=0.5): s.p, s.win = {}, window_s
    def update(s, t, tid, xy, rel=None, nearest_opp=None):
        """xy: pitch meters. rel: offset to own-team centroid already signed so +x = team's attacking direction. nearest_opp: distance (m) to closest opponent."""
        P = s.p.setdefault(tid, _P().init2()); x, y = xy
        if P.h and t - P.h[-1][0] > 0.8: P.h.clear(); P.v.clear(); P.in_sprint = False             # tracking gap -> new segment
        gy, gx = int(np.clip(y // CELL, 0, P.grid.shape[0] - 1)), int(np.clip(x // CELL, 0, P.grid.shape[1] - 1)); P.grid[gy, gx] += 1   # rows = y, cols = x
        if rel is not None: P.rel += rel; P.relN += 1; P.dnorm += float(np.hypot(*rel))
        ref = next(((t0, x0, y0) for t0, x0, y0 in reversed(P.h) if t - t0 >= s.win), None)           # sample >= 0.5 s ago -> noise-robust speed
        if ref is not None and t - ref[0] <= 1.2:
            v = float(np.hypot(x - ref[1], y - ref[2]) / (t - ref[0]))
            if v <= MAX_MS:
                if P.v: tv, vp = P.v[-1]; (P.acc.append((v - vp) / (t - tv)) if 0.15 <= t - tv <= 0.6 else None)
                P.v.append((t, v)); P.obs += (t - P.h[-1][0]) if P.h else 0; P.dist += v * ((t - P.h[-1][0]) if P.h else 0)
                recent = [vv for tt, vv in P.v if t - tt <= 1.0]; P.top = max(P.top, float(np.mean(recent)) if len(recent) >= 3 else 0)   # sustained 1 s speed
                P.v_sum += v; P.v_n += 1
                if nearest_opp is not None and nearest_opp < 4.0: P.pr_sum += v; P.pr_n += 1
                if v >= SPRINT_MS and not P.in_sprint: P.sprints += 1
                P.in_sprint = v >= SPRINT_MS
        P.h.append((t, x, y))
    @staticmethod
    def role(dx, dy_abs, team_attack_scale=1.0):
        """Role from the mean offset to the team centroid (dx: depth along attack, dy_abs: lateral). Thresholds in metres."""
        if dx < -22: return "GK", min(1.0, (-22 - dx) / 4)
        edges = [-8.0, 8.0]; line = "DEF" if dx < edges[0] else "FWD" if dx > edges[1] else "MID"; margin = min(abs(dx - e) for e in edges)
        wide = dy_abs > 17; name = {"DEF": "FB" if wide else "CB", "MID": "WM" if wide else "CM", "FWD": "W" if wide else "ST"}[line]
        return name, float(np.clip(margin / 4, 0, 1))
    def profile(s, tid, team=-1):
        P = s.p.get(tid)
        if P is None: return None
        r = P.rel / max(P.relN, 1); role, rc = s.role(r[0], abs(r[1])) if P.relN >= 5 else ("?", 0.0); mean_v = P.v_sum / max(P.v_n, 1)
        accp = np.array([a for a in P.acc if a > 0]); cells = int((P.grid > 0).sum()); pressured_ok = P.pr_n * 0.2 >= 2.0           # need ~2 s under pressure
        pr = float(np.clip((P.pr_sum / P.pr_n) / (mean_v + 1e-6), 0, 1.5) / 1.5) if pressured_ok else 0.5                        # mobility under pressure (0.5 = unknown)
        conf = float(min(1.0, P.obs / 20.0))
        return PlayerProfile(tid, team, role, rc * min(1.0, P.obs / 10), float(np.clip(P.top / 9.0, 0, 1)), float(np.clip((np.percentile(accp, 95) if len(accp) > 5 else 0) / 5.0, 0, 1)),
                             pr, float(np.clip(cells / 40.0, 0, 1)), float(np.clip(mean_v / 3.0, 0, 1)), P.top, P.sprints, P.dist, P.obs, conf)
    def heatmap(s, tid): P = s.p.get(tid); return None if P is None else P.grid.copy()
