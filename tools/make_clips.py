"""Cut a short clip around every key moment from the annotated video.  python -m tools.make_clips out.mp4 report.json clips/ [--pad 3]"""
import sys, json, os, subprocess, argparse
from analytics.pressing import key_moments
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("report"); ap.add_argument("outdir"); ap.add_argument("--pad", type=float, default=3.0); a = ap.parse_args()
R = json.load(open(a.report)); os.makedirs(a.outdir, exist_ok=True); ms = key_moments(R["timeline"])
for i, m in enumerate(ms):
    o = os.path.join(a.outdir, f"moment_{i+1:02d}_{m['tags'][0].lower()}_{m['t']:.0f}s.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(max(m["t"] - a.pad, 0)), "-t", str(2 * a.pad), "-i", a.video, "-c:v", "libx264", "-pix_fmt", "yuv420p", o], check=True)
    print(o, "-", m["why"])
json.dump(ms, open(os.path.join(a.outdir, "moments.json"), "w"), indent=1); print(len(ms), "clips")
