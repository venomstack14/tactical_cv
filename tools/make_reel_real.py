"""1080x1920 reel from an analysed real clip: auto-reframed hero panel following the players + full-frame panel + live counters + shot timeline.
python -m tools.make_reel_real annot.mp4 stats.json reel_real.mp4 [--repo github.com/you/tactical_cv] [--start 0 --end 28]"""
import json, argparse, os, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw
from tools.make_reel import F, W, H, text, ease
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("stats"); ap.add_argument("out"); ap.add_argument("--repo", default="github.com/venomstack14/tactical_cv"); ap.add_argument("--start", type=float, default=0); ap.add_argument("--end", type=float, default=1e9); a = ap.parse_args()
    S = json.load(open(a.stats)); fps = S["fps"]; rows = S["frames"]; shots = S["shots"]; dur = S["duration_s"]; cap = cv2.VideoCapture(a.video); sw, sh = S["size"]
    f0, f1 = int(a.start * fps), min(int(a.end * fps), len(rows)); cap.set(cv2.CAP_PROP_POS_FRAMES, f0); tmp = a.out + ".raw.mp4"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    yy = np.linspace(0, 1, H)[:, None, None]; bg = (np.array([28, 14, 10]) * (1 - yy) + np.array([10, 6, 5]) * yy).astype(np.uint8); bg = np.repeat(bg, W, 1)
    HX0, HY0, HW, HH = 0, 270, W, 860; cx, cur = sw / 2, -1; n = f1 - f0; END = int(2.0 * fps); STAT = int(5.0 * fps); fast = lambda v, k: v * ease(k); col = {"wide": (90, 220, 120), "close-up/other": (150, 150, 150), "non-pitch (crowd/graphic)": (80, 80, 80)}
    wide = [x for x in shots if x["type"] == "wide" and x.get("unique_track_ids")]; trk = S["tracks"]; tcnt = [sum(1 for t in trk if t["team"] == q) for q in (0, 1, -1)]; wsec = S["wide_seconds"]
    bw = sum(x["ball_visible_pct"] * (x["end_s"] - x["start_s"]) for x in wide) / max(sum(x["end_s"] - x["start_s"] for x in wide), 1e-9); rid = S.get("reid") or {}
    tiles = [("CLIP LENGTH", dur, "s", 1), ("TRACKED (WIDE SHOTS)", wsec, "s", 1), ("SHOTS DETECTED", len(shots), "", 0), ("MAX PLAYERS AT ONCE", max(x["max_simultaneous_players"] for x in wide), "", 0),
             ("PLAYER IDENTITIES", sum(x["unique_track_ids"] for x in wide), "", 0), ("BALL SEEN", bw, "%", 0), ("AVG TRACK LENGTH", float(np.mean([x["mean_track_seconds"] for x in wide])), "s", 1), ("TRACKS MERGED (RE-ID)", None, "", 0)]
    for k in range(n + STAT + END):
        i = min(f0 + k, f1 - 1)
        if k < n: ok, fr = cap.read(); last = fr
        else: fr = last
        if n <= k < n + STAT: pass
        r = rows[i]; im = bg.copy()
        if r["shot"] != cur: cur = r["shot"]; cx = None
        xs = [(t[2] + t[4]) / 2 for t in r["tracks"]]; tgt = float(np.median(xs)) if xs else sw / 2; cx = tgt if cx is None else 0.88 * cx + 0.12 * tgt
        cw = int(sh * HW / HH); x0 = int(np.clip(cx - cw / 2, 0, sw - cw)); crop = cv2.resize(fr[:, x0:x0 + cw], (HW, HH), interpolation=cv2.INTER_CUBIC); im[HY0:HY0 + HH] = crop
        sm = cv2.resize(fr, (W, int(W * sh / sw))); y1 = 1200; im[y1:y1 + sm.shape[0]] = sm; cv2.rectangle(im, (int(x0 * W / sw), y1), (int((x0 + cw) * W / sw), y1 + sm.shape[0]), (255, 255, 255), 3)      # white box = what the hero panel shows
        by = y1 + sm.shape[0] + 40; t0, t1 = rows[f0]["t"], rows[f1 - 1]["t"]; cv2.rectangle(im, (70, by), (W - 70, by + 22), (40, 40, 40), -1)
        for s in shots:
            a0, a1 = max(s["start_s"], t0), min(s["end_s"], t1)
            if a1 > a0: cv2.rectangle(im, (int(70 + (a0 - t0) / (t1 - t0) * (W - 140)), by), (int(70 + (a1 - t0) / (t1 - t0) * (W - 140)), by + 22), col.get(s["type"], (90, 90, 90)), -1)
        px = int(70 + (r["t"] - t0) / (t1 - t0) * (W - 140)); cv2.rectangle(im, (px - 3, by - 8), (px + 3, by + 30), (255, 255, 255), -1)
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).convert("RGBA")
        pil = text(pil, (W // 2, 120), "TACTICAL CV", F["h1"], (255, 255, 255), "mm"); pil = text(pil, (W // 2, 200), "REAL BROADCAST FOOTAGE  ·  AUTO TRACKING", F["small"], (170, 190, 210), "mm")
        wide = r["type"] == "wide"; n0, n1, n2 = r["n"]
        pil = text(pil, (60, HY0 + HH - 120), ("TRACKING  %d players" % sum(r["n"])) if wide else "REPLAY / CLOSE-UP - tracking paused", F["h2"], (120, 230, 140) if wide else (190, 190, 190), "la")
        if wide: pil = text(pil, (60, HY0 + HH - 68), "white kit %d  ·  dark kit %d  ·  other %d  ·  ball %s" % (n0, n1, n2, "yes" if r["ball"] else "–"), F["small"], (255, 255, 255), "la")
        pil = text(pil, (70, by + 52), "shot timeline:  green = wide (tracked)   grey = close-up / crowd", F["small"], (150, 165, 185), "la")
        if n <= k < n + STAT or k >= n + STAT:
            ks = ease((k - n) / (0.5 * fps)); ov = Image.new("RGBA", pil.size, (8, 12, 20, int(255 * ks))); pil = Image.alpha_composite(pil, ov)
        if n <= k < n + STAT:
            T_ = (k - n) / fps; pil = text(pil, (W // 2, 300), "CLIP STATS", F["h1"], (255, 255, 255), "mm", ks); pil = text(pil, (W // 2, 380), "measured by the tracker on this footage", F["small"], (170, 190, 210), "mm", ks)
            ov = Image.new("RGBA", pil.size, (0, 0, 0, 0)); dd = ImageDraw.Draw(ov)
            for q, (lab, val, unit, dec) in enumerate(tiles):
                x0, y0 = 60 + (q % 2) * 500, 450 + (q // 2) * 215; ap_ = ease((T_ - 0.15 * q) / 0.5); dd.rounded_rectangle((x0, y0, x0 + 460, y0 + 190), 26, fill=(22, 30, 46, int(235 * ap_)), outline=(60, 80, 110, int(255 * ap_)), width=2)
            pil = Image.alpha_composite(pil, ov)
            for q, (lab, val, unit, dec) in enumerate(tiles):
                x0, y0 = 60 + (q % 2) * 500, 450 + (q // 2) * 215; ap_ = ease((T_ - 0.15 * q) / 0.5)
                txt_ = ("%d identities" % rid["identities"]) if val is None and rid else ("%.*f%s" % (dec, val * ease((T_ - 0.15 * q) / 1.2), unit)) if val is not None else "-"
                if val is None and rid: txt_ = "%d to %d" % (rid["tracklets"], rid["identities"])
                pil = text(pil, (x0 + 30, y0 + 50), txt_, F["chip"] if len(txt_) < 9 else F["h2"], (80, 230, 255), "la", ap_); pil = text(pil, (x0 + 30, y0 + 132), lab, F["small"], (150, 165, 185), "la", ap_)
            tot = max(sum(tcnt), 1); bx, by_, bw_ = 60, 1350, 960; ap_ = ease((T_ - 1.4) / 0.6); ov = Image.new("RGBA", pil.size, (0, 0, 0, 0)); dd = ImageDraw.Draw(ov); xx = bx
            for cnt, colr in zip(tcnt, ((235, 235, 235), (60, 130, 255), (120, 120, 120))):
                w_ = int(bw_ * cnt / tot * min(ap_ * 1.5, 1)); dd.rounded_rectangle((xx, by_, xx + max(w_, 2), by_ + 46), 10, fill=(*colr, int(255 * ap_))); xx += w_
            pil = Image.alpha_composite(pil, ov); pil = text(pil, (bx, by_ - 50), "TRACKED PLAYERS BY KIT", F["small"], (150, 165, 185), "la", ap_); pil = text(pil, (bx, by_ + 64), "white kit %d   ·   dark kit %d   ·   officials/unassigned %d" % tuple(tcnt), F["small"], (255, 255, 255), "la", ap_)
            pil = text(pil, (W // 2, 1560), "Image-space tracking in wide shots only.", F["small"], (150, 165, 185), "mm", ap_); pil = text(pil, (W // 2, 1605), "No distances or speeds: the camera pans and the footage is edited.", F["small"], (150, 165, 185), "mm", ap_)
        if k >= n + STAT:
            kk = ease((k - n - STAT) / (0.5 * fps)); ov = Image.new("RGBA", pil.size, (8, 12, 20, int(255 * kk))); pil = Image.alpha_composite(pil, ov)
            pil = text(pil, (W // 2, 800), "OPEN SOURCE", F["h1"], (255, 255, 255), "mm", kk); pil = text(pil, (W // 2, 920), "Tracks players from raw broadcast video", F["h2"], (80, 230, 255), "mm", kk); pil = text(pil, (W // 2, 1000), "CPU only · optional NVIDIA NIM", F["body"], (170, 190, 210), "mm", kk); pil = text(pil, (W // 2, 1140), "★  " + a.repo, F["h2"], (255, 255, 255), "mm", kk)
        vw.write(cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR))
    vw.release(); subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True); os.remove(tmp); print("wrote", a.out)
if __name__ == "__main__": main()
