"""Dynamic camera calibration: broadcast pixels -> 105x68 m pitch plane.

Pipeline
 1. INIT   : >=4 known pitch points (JSON / mouse clicks / a YOLOv8-pose keypoint model) -> cv2.findHomography(RANSAC).
 2. TRACK  : every frame, Lucas-Kanade tracks background features on the pitch (players masked out); a frame-to-frame
             homography M is fitted with RANSAC and the transform is propagated:  H_t = H_{t-1} . M^-1 .
 3. REFINE : every N frames the detected white-line mask is warped to the pitch plane and aligned to the ideal line
             template with ECC (homography motion) - this removes the drift that pure propagation accumulates.
 4. GATE   : low inlier counts / implausible corrections never silently update H (state.valid=False)."""
import json, cv2, numpy as np
from dataclasses import dataclass
from core import pitch

@dataclass
class CalibState:
    H: np.ndarray          # 3x3, processing-frame pixels -> pitch meters
    valid: bool            # False when the camera motion could not be estimated this frame (H is held)
    quality: float         # fraction of tracked features that were RANSAC inliers
    refined: bool = False  # True on frames where line-ECC refinement was applied

class PitchSegmenter:
    """Classical pitch/line segmentation (HSV green + white top-hat) with Hough segments and their intersections."""
    def __init__(s, lo=(30, 40, 30), hi=(90, 255, 255)): s.lo, s.hi = np.array(lo, np.uint8), np.array(hi, np.uint8)
    def pitch_mask(s, bgr): return cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), s.lo, s.hi)
    @staticmethod
    def fill(mask, k=25):  # close holes made by lines/players so the pitch becomes one solid region
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    def line_mask(s, bgr, filled=None):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        m = ((th > 35) * 255).astype(np.uint8)                                         # thin bright structures
        filled = s.fill(s.pitch_mask(bgr)) if filled is None else filled
        return cv2.bitwise_and(m, filled)
    @staticmethod
    def segments(lmask):
        v = cv2.HoughLinesP(lmask, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=12); return np.zeros((0, 4)) if v is None else v[:, 0, :].astype(float)
    @staticmethod
    def intersections(segs, shape, tol=20.0):
        """Intersections of extended segments that fall inside the image and near both segments (corner / junction candidates)."""
        out = []; h, w = shape[:2]
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                (x1, y1, x2, y2), (x3, y3, x4, y4) = segs[i], segs[j]; d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(d) < 1e-6: continue
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d; py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
                near = lambda a, b, c, e: min(a, c) - tol <= px <= max(a, c) + tol and min(b, e) - tol <= py <= max(b, e) + tol
                if 0 <= px < w and 0 <= py < h and near(x1, y1, x2, y2) and near(x3, y3, x4, y4): out.append((px, py))
        return np.array(out)

def load_correspondences(path):
    """JSON: [{"px":[x,y], "name":"center_spot"} | {"px":[x,y], "pitch":[X,Y]}, ...] in PROCESSING-frame pixels."""
    out = []
    for c in json.load(open(path)): out.append((c["px"], pitch.LANDMARKS[c["name"]] if "name" in c else c["pitch"]))
    return out
def solve_from_keypoints(img_pts, pitch_pts, min_pts=4, max_err_m=1.5):
    """RANSAC homography from detected keypoints (e.g. a YOLOv8-pose model). Returns (H|None, reason)."""
    img, wld = np.float32(img_pts).reshape(-1, 2), np.float32(pitch_pts).reshape(-1, 2)
    if len(img) < min_pts: return None, "too_few_points"
    H, m = cv2.findHomography(img, wld, cv2.RANSAC if len(img) > 4 else 0, 1.5)
    if H is None: return None, "solve_failed"
    ok = np.ones(len(img), bool) if m is None else m.ravel() == 1
    err = np.linalg.norm(cv2.perspectiveTransform(img[ok].reshape(-1, 1, 2), H).reshape(-1, 2) - wld[ok], axis=1).mean()
    return (H / H[2, 2], "ok") if err <= max_err_m else (None, "high_reproj")

class PitchCalibrator:
    PPM = 4  # raster pixels per meter for the line-ECC refinement
    def __init__(s, frame_size, work_width=480, refine_every=15, min_inliers=18, max_correction_m=2.5):
        w, h = frame_size; s.sc = work_width / w; s.work = (work_width, int(round(h * s.sc))); s.S = np.diag([s.sc, s.sc, 1.0])
        s.H, s.prev, s.prev_mask, s.n, s.seg = None, None, None, 0, PitchSegmenter(); s.refine_every, s.min_inliers, s.max_corr = refine_every, min_inliers, max_correction_m
        s.tpl = pitch.template_raster(s.PPM)
    def init_from_correspondences(s, corr):
        r = solve_from_keypoints([c[0] for c in corr], [c[1] for c in corr], max_err_m=3.0)
        if r[0] is None: raise ValueError(f"calibration failed: {r[1]}")
        s.H = r[0]; return s.H
    def to_pitch(s, pts_px, H=None):
        p = np.float32(pts_px).reshape(-1, 1, 2); return cv2.perspectiveTransform(p, s.H if H is None else H).reshape(-1, 2) if len(p) else np.zeros((0, 2))
    def update(s, frame, boxes=None):
        """frame: processing-size BGR image. boxes: Nx4 player boxes to exclude from feature tracking."""
        small = cv2.resize(frame, s.work, interpolation=cv2.INTER_AREA); gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = cv2.erode(s.seg.fill(s.seg.pitch_mask(small), 15), np.ones((5, 5), np.uint8))
        for b in (boxes if boxes is not None else []): x1, y1, x2, y2 = (np.asarray(b) * s.sc).astype(int); mask[max(y1 - 4, 0):y2 + 4, max(x1 - 4, 0):x2 + 4] = 0
        quality, valid = 0.0, False
        if s.prev is not None:
            p0 = cv2.goodFeaturesToTrack(s.prev, 300, 0.01, 7, mask=s.prev_mask)
            if p0 is not None and len(p0) >= s.min_inliers:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(s.prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1; a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
                if len(a) >= s.min_inliers:
                    M, inl = cv2.findHomography(a, b, cv2.RANSAC, 1.5)
                    if M is not None and inl.sum() >= s.min_inliers and np.abs(cv2.perspectiveTransform(np.float32([[[0, 0], [work_w := s.work[0], s.work[1]]]]), M) - np.float32([[[0, 0], [work_w, s.work[1]]]])).max() < 60:
                        s.H = s.H @ np.linalg.inv(np.linalg.inv(s.S) @ M @ s.S); s.H /= s.H[2, 2]; quality, valid = float(inl.mean()), True
        s.prev, s.prev_mask, s.n = gray, mask, s.n + 1; refined = False
        if valid and s.n % s.refine_every == 0: refined = s.refine_with_lines(frame)
        return CalibState(s.H.copy(), valid, quality, refined)
    def refine_with_lines(s, frame):
        """Align the detected line mask (warped to the pitch plane) to the ideal template with ECC. Returns True if applied."""
        lm = s.seg.line_mask(frame).astype(np.float32) / 255; S = np.diag([s.PPM, s.PPM, 1.0]); size = (int(pitch.L * s.PPM), int(pitch.W * s.PPM))
        Lw = cv2.warpPerspective(lm, S @ s.H, size); V = (cv2.warpPerspective(np.ones_like(lm), S @ s.H, size) > 0.99).astype(np.float32)
        if V.mean() < 0.06 or Lw.sum() < 60: return False
        T = s.tpl * V                                                                   # compare only where the camera sees the pitch
        Wm = np.eye(3, dtype=np.float32)
        try: cc, Wm = cv2.findTransformECC(cv2.GaussianBlur(T, (0, 0), 2), cv2.GaussianBlur(Lw, (0, 0), 2), Wm, cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        except cv2.error: return False
        Hn = np.linalg.inv(S) @ np.linalg.inv(Wm) @ S @ s.H; Hn /= Hn[2, 2]
        g = np.array([[x, y, 1.0] for x in range(20, 90, 10) for y in range(10, 60, 10)]).T   # correction magnitude over the pitch
        a, b = s.H @ np.linalg.inv(s.H) @ g, Hn @ np.linalg.inv(s.H) @ g; d = np.linalg.norm(b[:2] / b[2] - a[:2] / a[2], axis=0)
        if cc < 0.2 or np.median(d) > s.max_corr: return False
        s.H = Hn; return True

def interactive_calibration(frame, names=("center_spot", "circle_near", "circle_far", "halfway_near", "halfway_far", "left_box_near", "left_box_far", "right_box_near", "right_box_far")):
    """Mouse init on the first frame: click each requested landmark (press 'n' to skip one, 'q' to finish once >=4 are set). Needs a display."""
    pts, i, win = [], 0, "calibration: click the landmark named in the title"
    def cb(ev, x, y, *_):
        nonlocal i
        if ev == cv2.EVENT_LBUTTONDOWN and i < len(names): pts.append(((x, y), pitch.LANDMARKS[names[i]])); i += 1
    cv2.namedWindow(win); cv2.setMouseCallback(win, cb)
    while i < len(names):
        img = frame.copy()
        for (px, _) in pts: cv2.circle(img, (int(px[0]), int(px[1])), 5, (0, 0, 255), -1)
        cv2.putText(img, f"click: {names[i]}   [n]=skip  [q]=done ({len(pts)} set)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); cv2.imshow(win, img); k = cv2.waitKey(30) & 0xFF
        if k == ord("n"): i += 1
        if k == ord("q") and len(pts) >= 4: break
    cv2.destroyWindow(win); return pts
