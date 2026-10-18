"""Synthetic broadcast-style clip with KNOWN ground truth (camera homography + player positions) for testing.
python -m tools.make_demo_video demo.mp4 [--seconds 8]  -> demo.mp4, demo_truth.json (+ demo_calib.json for the CLI)."""
import sys, json, cv2, numpy as np
from core import pitch
W_, H_, FPS, F = 960, 540, 25, 700.0
K = np.array([[F, 0, W_ / 2], [0, F, H_ / 2], [0, 0, 1.0]])
def camera(t, dur):
    """Pan across the pitch: returns (Rot, C) for a camera on the near touchline, 22 m high."""
    cx = 38 + 24 * (t / dur); C = np.array([cx - 6, -38.0, 22.0]); f = np.array([cx, 34, 0]) - C; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f); return np.stack([r, -u, f]), C
def H_pitch2img(R, C): t = -R @ C; return K @ np.stack([R[:, 0], R[:, 1], t], 1)
def project3(R, C, X): p = K @ (R @ (np.asarray(X, float) - C)); return p[:2] / p[2]
def texture(ppm=10, margin=12):
    h, w = int((pitch.W + 2 * margin) * ppm), int((pitch.L + 2 * margin) * ppm); rng = np.random.RandomState(1)
    img = np.zeros((h, w, 3), np.uint8); img[:] = (58, 128, 52)
    for i in range(0, w, 7 * ppm): img[:, i:i + 7 * ppm // 2] = (52, 118, 46)                  # mowing stripes
    img = np.clip(img + rng.randint(-14, 14, img.shape), 0, 255).astype(np.uint8)
    for p in pitch.polylines(): cv2.polylines(img, [np.round((p + margin) * ppm).astype(np.int32)], False, (235, 235, 235), 3)
    return img, ppm, margin
TEX = None
def players_at(t, dur):
    """Truth: red = 'our' team attacking +x, blue = opponent block whose back line rises from 28 m to 40 m from its goal."""
    h = 28 + 12 * (t / dur); ph = np.arange(22) * 1.7; out = {}
    ours = [(5, 34)] + [(30, y) for y in (12, 26, 42, 56)] + [(45, y) for y in (18, 34, 50)] + [(62, y) for y in (10, 34, 58)]
    opp = [(104, 34)] + [(105 - h, y) for y in (10, 25, 43, 58)] + [(105 - h - 14, y) for y in (16, 34, 52)] + [(105 - h - 26, y) for y in (20, 34, 48)]
    for i, (x, y) in enumerate(ours): out[i] = ("A", x + 2.5 * np.sin(0.7 * t + ph[i]) + 6 * t / dur, y + 2 * np.cos(0.5 * t + ph[i]))
    for i, (x, y) in enumerate(opp): out[11 + i] = ("B", x + 1.5 * np.sin(0.6 * t + ph[11 + i]), y + 2 * np.cos(0.4 * t + ph[11 + i]))
    return out
def render(t, dur):
    global TEX
    TEX = TEX or texture(); tex, ppm, mg = TEX; R, C = camera(t, dur); Hp = H_pitch2img(R, C); Sx = np.diag([ppm, ppm, 1.0]); T = np.diag([1, 1, 1.0]); T[:2, 2] = mg
    img = cv2.warpPerspective(tex, Hp @ np.linalg.inv(Sx @ T), (W_, H_), borderMode=cv2.BORDER_CONSTANT, borderValue=(70, 62, 60))
    truth = {}
    for pid, (team, x, y) in sorted(players_at(t, dur).items(), key=lambda kv: -kv[1][2]):    # far players first
        foot, head = project3(R, C, (x, y, 0)), project3(R, C, (x, y, 1.8)); hpx = foot[1] - head[1]; wpx = hpx * 0.38; col = (40, 40, 215) if team == "A" else (215, 70, 40)
        x1, y1, x2, y2 = int(foot[0] - wpx / 2), int(head[1]), int(foot[0] + wpx / 2), int(foot[1])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, -1); cv2.circle(img, (int(foot[0]), int(head[1]) - 3), max(int(wpx * .3), 2), (170, 190, 220), -1)
        truth[pid] = {"team": team, "xy": [x, y], "foot_px": foot.tolist(), "box": [x1, y1, x2, y2]}
    return img, Hp, truth
def main(out, seconds=8):
    n = int(seconds * FPS); vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W_, H_)); truth = []
    for i in range(n): img, Hp, tr = render(i / FPS, seconds); vw.write(img); truth.append({"H": Hp.tolist(), "players": tr})
        
    vw.release(); json.dump(truth, open(out.rsplit(".", 1)[0] + "_truth.json", "w"))
    Hp = np.array(truth[0]["H"]); names = ["center_spot", "circle_near", "circle_far", "halfway_near", "halfway_far", "left_box_far", "left_box_near", "right_box_near"]
    corr = []
    for nm in names:
        p = cv2.perspectiveTransform(np.float32([[pitch.LANDMARKS[nm]]]), Hp)[0, 0]
        if 5 < p[0] < W_ - 5 and 5 < p[1] < H_ - 5: corr.append({"px": (p + np.random.RandomState(2).randn(2)).round(1).tolist(), "name": nm})
    json.dump(corr, open(out.rsplit(".", 1)[0] + "_calib.json", "w"), indent=1); print("wrote", out, len(corr), "calibration points")
if __name__ == "__main__": main(sys.argv[1], float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 8)
