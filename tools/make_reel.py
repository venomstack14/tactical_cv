"""Vertical 1080x1920 (Instagram Reels / TikTok / Shorts) animated tactical replay rendered from report.json - no source video needed, so it is safe to post.
python -m tools.make_reel report.json reel.mp4 [--repo github.com/you/tactical_cv] [--title "PRESS RESISTANCE"]"""
import json, argparse, os, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from core import pitch
W, H, FPS = 1080, 1920, 30; PX0, PX1, PY0, PY1 = 70, 1010, 330, 1560                    # pitch box (portrait: pitch length runs bottom->top)
A = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
def font(sz, bold=True):
    for p in (os.path.join(A, "Poppins-Bold.ttf" if bold else "Poppins-Medium.ttf"), "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()
F = {k: font(v, b) for k, (v, b) in dict(h1=(86, 1), h2=(40, 1), body=(34, 0), chip=(46, 1), small=(28, 0)).items()}
def P(x, y): return int(PX0 + y / pitch.W * (PX1 - PX0)), int(PY1 - x / pitch.L * (PY1 - PY0))
def hexbgr(h): return (int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16))
def bright(c):
    hsv = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2HSV)[0, 0].astype(float); hsv[1] = max(hsv[1], 170); hsv[2] = max(hsv[2], 235); return tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0])
def ease(t): t = min(max(t, 0), 1); return t * t * (3 - 2 * t)
def base():
    img = np.zeros((H, W, 3), np.uint8); yy = np.linspace(0, 1, H)[:, None, None]; img[:] = (np.array([28, 14, 10]) * (1 - yy) + np.array([10, 6, 5]) * yy).astype(np.uint8)   # deep navy gradient (BGR)
    g = np.zeros_like(img); cv2.rectangle(g, (PX0, PY0), (PX1, PY1), (70, 70, 25), -1)
    for i in range(0, 12):                                                                      # mowing stripes
        y0 = PY0 + i * (PY1 - PY0) // 12; cv2.rectangle(g, (PX0, y0), (PX1, y0 + (PY1 - PY0) // 12), (62, 62, 20) if i % 2 else (78, 78, 28), -1)
    m = np.zeros((H, W), np.uint8); cv2.rectangle(m, (PX0, PY0), (PX1, PY1), 255, -1); img[m > 0] = cv2.addWeighted(img, .25, g, .75, 0)[m > 0]
    lines = np.zeros_like(img)
    for pl in pitch.polylines(): cv2.polylines(lines, [np.array([P(*p) for p in pl], np.int32)], False, (255, 245, 200), 3, cv2.LINE_AA)
    glow = cv2.GaussianBlur(lines, (0, 0), 9); return cv2.add(cv2.add(img, (glow * .9).astype(np.uint8)), (lines * .55).astype(np.uint8))
def series(frames):
    S = {}
    for f in frames:
        for pid, tm, x, y in f["p"]: S.setdefault(pid, []).append((f["t"], tm, x, y))
    return S
def at(S, t, maxgap=0.45):
    out = {}
    for pid, v in S.items():
        ts = [a[0] for a in v]; i = np.searchsorted(ts, t)
        if i == 0 or i >= len(v):
            if v and abs(v[min(i, len(v) - 1)][0] - t) < 0.15: a = v[min(i, len(v) - 1)]; out[pid] = (a[1], a[2], a[3])
            continue
        a, b = v[i - 1], v[i]
        if b[0] - a[0] > maxgap: continue
        k = (t - a[0]) / (b[0] - a[0]); out[pid] = (b[1], a[2] + k * (b[2] - a[2]), a[3] + k * (b[3] - a[3]))
    return out
def text(im, xy, s, f, col=(255, 255, 255), anchor="la", alpha=1.0):
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov); d.text(xy, s, font=f, fill=(*col[::-1], int(255 * alpha)), anchor=anchor); return Image.alpha_composite(im, ov)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("report"); ap.add_argument("out"); ap.add_argument("--repo", default="github.com/venomstack14/tactical_cv"); ap.add_argument("--title", default="TACTICAL CV"); ap.add_argument("--club", default="")
    a = ap.parse_args(); R = json.load(open(a.report)); S = series(R["frames"]); our = R["our"]; d = R["attack"]; cols = {int(k): hexbgr(v) for k, v in R["team_colors"].items()}; cols = {k: bright(v) for k, v in cols.items()}
    dur = R["duration"]; INTRO, OUTRO = 1.4, 2.2; total = INTRO + dur + OUTRO; bg = base(); tl = R["timeline"]; trail = {}
    tmp = a.out + ".raw.mp4"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)); prm = (R.get("pressing") or {}).get("ours", {}).get("press_index")
    for fi in range(int(total * FPS)):
        T = fi / FPS; t = T - INTRO; fr = bg.copy(); lay = np.zeros_like(fr)
        if 0 <= t <= dur + .01:
            pos = at(S, min(t, dur)); 
            for pid, (tm, x, y) in pos.items(): trail.setdefault(pid, []).append(P(x, y)); trail[pid] = trail[pid][-45:]
            opp = np.array([P(x, y) for pid, (tm, x, y) in pos.items() if tm >= 0 and tm != our])
            if len(opp) >= 3: h = cv2.convexHull(opp.astype(np.int32)); cv2.fillConvexPoly(lay, h, tuple(int(c * .55) for c in cols.get(1 - our, (200, 200, 200)))); cv2.polylines(lay, [h], True, cols.get(1 - our, (200, 200, 200)), 4, cv2.LINE_AA)
            ox = sorted([x * d for pid, (tm, x, y) in pos.items() if tm >= 0 and tm != our], reverse=True)       # opp defensive line: 4 deepest outfielders (GK skipped)
            if len(ox) >= 6: lx = np.mean(ox[1:5]) * d; p0, p1 = P(lx, 0), P(lx, pitch.W); cv2.line(lay, p0, p1, (60, 220, 255), 5, cv2.LINE_AA); cv2.putText(lay, "", p0, 0, 1, (0, 0, 0))
            for pid, tr in trail.items():
                tm = pos.get(pid, (-1,))[0]
                if pid in pos and len(tr) > 2:
                    c = cols.get(tm, (190, 190, 190))
                    for k in range(1, len(tr)): cv2.line(lay, tr[k - 1], tr[k], tuple(int(v * k / len(tr)) for v in c), 6, cv2.LINE_AA)
            for pid, (tm, x, y) in pos.items():
                if tm < 0: continue
                c = cols.get(tm, (190, 190, 190)); p = P(x, y); cv2.circle(lay, p, 24, c, -1, cv2.LINE_AA)
                if tm == our: cv2.circle(lay, p, 29, (255, 255, 255), 3, cv2.LINE_AA)
            glow = cv2.GaussianBlur(lay, (0, 0), 14); fr = cv2.add(cv2.add(fr, glow), lay)
        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        im = text(im, (W // 2, 150), a.title, F["h1"], (255, 255, 255), "mm"); im = text(im, (W // 2, 232), "OPPOSITION ANALYSIS  ·  BROADCAST FOOTAGE  ·  CPU ONLY" if not a.club else a.club.upper(), F["small"], (170, 190, 210), "mm")
        if 0 <= t <= dur:                                                                         # live HUD chips + caption
            M = next((e for e in reversed(tl) if e["t"] <= t), None); mm = (M or {}).get("m") or {}
            chips = [("LINE", f'{mm["line_height"]:.0f} m') if "line_height" in mm else None, ("WIDTH", f'{mm["width"]:.0f} m') if "width" in mm else None, ("COMPACT", f'{mm["compactness"]:.2f}') if "compactness" in mm else None, ("PRESS IDX", f"{prm:.2f}") if prm is not None else None]
            for i, c in enumerate(filter(None, chips)): x0 = 70 + i * 235; im = text(im, (x0, 1610), c[1], F["chip"], (80, 230, 255)); im = text(im, (x0, 1668), c[0], F["small"], (150, 165, 185))
            rc = (M or {}).get("r") or []
            if rc:
                msg = rc[0][2].split(":")[0]; col = {1: (90, 90, 255), 2: (80, 170, 255)}.get(rc[0][0], (210, 210, 210)); al = ease((t - M["t"]) / 0.4)
                ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); dd = ImageDraw.Draw(ov); dd.rounded_rectangle((60, 1730, W - 60, 1860), 28, fill=(14, 20, 32, int(215 * al)), outline=(*col[::-1], int(255 * al)), width=4); im = Image.alpha_composite(im, ov)
                words, lines, cur = msg.split(), [], ""
                for w_ in words:
                    if len(cur) + len(w_) > 44: lines.append(cur); cur = w_
                    else: cur = (cur + " " + w_).strip()
                lines.append(cur)
                im = text(im, (90, 1746), lines[0], F["h2"], (255, 255, 255), "la", al); ev = (rc[0][3] or "")[:60]; im = text(im, (90, 1802), "evidence: " + ev if ev else "", F["small"], (150, 170, 195), "la", al)
        if T < INTRO:
            ov = Image.new("RGBA", im.size, (8, 12, 20, int(255 * (1 - ease(T / INTRO) * .9)))); im = Image.alpha_composite(im, ov)
        if t > dur:
            k = ease((t - dur) / 0.6); ov = Image.new("RGBA", im.size, (8, 12, 20, int(235 * k))); im = Image.alpha_composite(im, ov)
            im = text(im, (W // 2, 760), "OPEN SOURCE", F["h1"], (255, 255, 255), "mm", k); im = text(im, (W // 2, 880), "Football tactics from raw broadcast video", F["h2"], (80, 230, 255), "mm", k); im = text(im, (W // 2, 950), "No GPU · No cloud required · Optional NVIDIA NIM", F["body"], (170, 190, 210), "mm", k)
            im = text(im, (W // 2, 1120), "★  " + a.repo, F["h2"], (255, 255, 255), "mm", k)
        vw.write(cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR))
    vw.release(); subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True); os.remove(tmp); print("wrote", a.out)
if __name__ == "__main__": main()
