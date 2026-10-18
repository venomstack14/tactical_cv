"""Rendering: annotated video (left) + top-down radar and tactical panel (right)."""
import cv2, numpy as np
from core import pitch
FONT = cv2.FONT_HERSHEY_SIMPLEX; PRI_COL = {1: (60, 60, 255), 2: (0, 165, 255), 3: (200, 200, 200)}
def txt(img, s, org, scale=0.45, col=(255, 255, 255), th=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA); cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)
def headline(rec, n=70): s = rec.message.split(":")[0]; return (s if len(s) <= n else s[:n - 1] + "…")
def draw_video(frame, tracks, info, team_bgr, recs, fps, our_team):
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t.box.astype(int); col = team_bgr(t.team) if t.team >= 0 else (180, 180, 180); cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        txt(out, f"{t.id}{'*' if t.team == our_team else ''}", (x1, max(y1 - 4, 10)), 0.4, col)
    y = 22
    for r in recs[:3]: txt(out, headline(r), (10, y), 0.55, PRI_COL[r.priority], 2); y += 22
    txt(out, f"{fps:4.1f} FPS | calib {'OK' if info['calib_valid'] else 'HOLD'} q={info['calib_q']:.2f}{' +lines' if info['refined'] else ''}", (10, out.shape[0] - 10), 0.5, (0, 255, 255)); return out
def draw_radar(size, players, ball, team_bgr, our_team):
    """players: list of (id, team, x, y) in metres. Pitch y axis points away from the main camera, so y is drawn upward."""
    w, h = size; m = 12; sx, sy = (w - 2 * m) / pitch.L, (h - 2 * m) / pitch.W; img = np.full((h, w, 3), (45, 110, 50), np.uint8)
    P = lambda x, y: (int(m + x * sx), int(h - m - y * sy))
    for pl in pitch.polylines(): cv2.polylines(img, [np.array([P(*p) for p in pl], np.int32)], False, (230, 230, 230), 1, cv2.LINE_AA)
    for pid, team, x, y in players:
        c = team_bgr(team) if team >= 0 else (170, 170, 170); cv2.circle(img, P(x, y), 6, c, -1); cv2.circle(img, P(x, y), 6, (255, 255, 255) if team == our_team else (0, 0, 0), 1, cv2.LINE_AA); txt(img, str(pid), (P(x, y)[0] + 6, P(x, y)[1] - 4), 0.33)
    if ball: cv2.circle(img, P(*ball), 4, (255, 255, 255), -1)
    return img
def draw_panel(radar, metrics, recs, profiles, height):
    w = radar.shape[1]; panel = np.full((height, w, 3), (30, 30, 30), np.uint8); panel[:radar.shape[0]] = radar; y = radar.shape[0] + 18
    if metrics: txt(panel, f"Opp line {metrics['line_height']:.0f}m  width {metrics['width']:.0f}m  compact {metrics['compactness']:.2f}", (8, y), 0.45, (0, 255, 255)); y += 16; txt(panel, f"half-spaces opp L/R {metrics['opp_hs'][1]}/{metrics['opp_hs'][0]}  press {metrics['press_n']}", (8, y), 0.45, (0, 255, 255)); y += 20
    for r in recs[:4]:
        words, line = r.message.split(), ""
        for wd in words:
            if len(line) + len(wd) > 56: txt(panel, line, (8, y), 0.38, PRI_COL[r.priority]); y += 14; line = wd
            else: line = (line + " " + wd).strip()
        txt(panel, line, (8, y), 0.38, PRI_COL[r.priority]); y += 18
        if y > height - 58: break
    top = sorted(profiles, key=lambda p: -p.speed_index)[:3]
    for k, p in enumerate(top): txt(panel, f"#{p.id} {p.role:<2} spd {p.speed_index:.2f} acc {p.accel_index:.2f} prs {p.press_resistance:.2f} cov {p.pitch_coverage:.2f} c{p.confidence:.1f}", (8, height - 8 - 14 * (len(top) - 1 - k)), 0.36, (200, 255, 200))
    return panel
