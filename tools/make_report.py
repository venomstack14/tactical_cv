"""report.json -> one self-contained, club-presentable HTML scouting page (no external assets).  python -m tools.make_report report.json scouting.html [--club "Name"]"""
import json, sys, html, argparse, numpy as np
L, W = 105, 68; S = 8                                                              # svg scale px/m
def pitch_svg(inner, w=L * S, h=W * S):
    ln = 'stroke="#e8efe8" stroke-width="1.5" fill="none"'; X = lambda x: x * S; Y = lambda y: (W - y) * S
    return (f'<svg viewBox="0 0 {w} {h}" class="pitch"><rect width="{w}" height="{h}" fill="#2b6a3a"/><rect x="2" y="2" width="{w-4}" height="{h-4}" {ln}/><line x1="{w/2}" y1="2" x2="{w/2}" y2="{h-2}" {ln}/>'
            f'<circle cx="{w/2}" cy="{h/2}" r="{9.15*S}" {ln}/>' + "".join(f'<rect x="{x}" y="{Y(54.16/2+ 20.16)+0}" width="{16.5*S}" height="{40.32*S}" {ln}/>' for x in (2, w - 2 - 16.5 * S)) + inner + '</svg>')
def esc(s): return html.escape(str(s))
def build(R, club):
    fr = R["frames"]; our = R["our"]; tc = R["team_colors"]; col = lambda k: tc.get(str(k), "#888")
    avg = {}
    for f in fr:
        for pid, tm, x, y in f["p"]: avg.setdefault((pid, tm), []).append((x, y))
    dots = "".join(f'<circle cx="{np.mean([p[0] for p in v])*S:.0f}" cy="{(W-np.mean([p[1] for p in v]))*S:.0f}" r="9" fill="{col(tm)}" stroke="{"#fff" if tm==our else "#000"}" stroke-width="2"><title>#{pid} avg position ({len(v)} samples)</title></circle>'
                   f'<text x="{np.mean([p[0] for p in v])*S:.0f}" y="{(W-np.mean([p[1] for p in v]))*S+3:.0f}" font-size="9" text-anchor="middle" fill="#fff" font-weight="700">{pid}</text>' for (pid, tm), v in avg.items() if len(v) >= 5)
    xs = np.array([[x, y] for f in fr for _, tm, x, y in f["p"] if tm != our and tm >= 0]); heat = ""
    if len(xs):
        Hh, _, _ = np.histogram2d(xs[:, 0], xs[:, 1], bins=[21, 14], range=[[0, L], [0, W]]); m = Hh.max() or 1
        heat = "".join(f'<rect x="{i*L*S/21:.0f}" y="{(13-j)*W*S/14:.0f}" width="{L*S/21:.0f}" height="{W*S/14:.0f}" fill="#ffb300" opacity="{0.75*Hh[i,j]/m:.2f}"/>' for i in range(21) for j in range(14) if Hh[i, j] > 0)
    M = R.get("metrics") or {}; ms = R.get("match_stats") or {}; ts = R.get("team_stats") or {}
    tiles = [("Opp. line height", f'{M["line_height"]:.0f} m') if "line_height" in M else None, ("Opp. width", f'{M["width"]:.0f} m') if "width" in M else None, ("Compactness", f'{M["compactness"]:.2f}') if "compactness" in M else None,
             ("Opp. distance (observed)", f'{ts.get("opp",{}).get("distance_m","–")} m'), ("Opp. top speed", f'{ts.get("opp",{}).get("top_speed_kmh","–")} km/h'), ("Opp. sprints", ts.get("opp", {}).get("sprints", "–"))]
    tiles = "".join(f'<div class="tile"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for k, v in filter(None, tiles))
    recs = "".join(f'<li class="p{r["priority"]}"><b>{esc(r["message"])}</b><small>{esc(r["evidence"])}</small></li>' for r in R.get("recommendations", [])) or "<li>No rule fired (insufficient coverage or neutral shape).</li>"
    rows = "".join(f'<tr><td>#{p["id"]}{" · "+esc(p["identity"]["name"] or "no. "+str(p["identity"]["number"])) if p.get("identity") else ""}</td><td>{"ours" if p["team"]==our else "opp"}</td><td>{p["role"]}</td>'
                   + "".join(f'<td><i style="width:{int(100*v)}%"></i></td>' for v in p["vector"]) + f'<td>{p["confidence"]:.1f}</td></tr>' for p in sorted(R["profiles"], key=lambda p: (p["team"] != our, -p["confidence"]))[:24])
    ai = R.get("ai_brief"); ai = f'<section><h2>AI scouting brief <em>NVIDIA NIM · generated only from the numbers in this report</em></h2><pre>{esc(ai)}</pre></section>' if ai else ""
    nim = R.get("nim"); prov = f'NIM assist: {nim["calls"]} calls, {nim["errors"]} errors, {R.get("referees_filtered",0)} officials/staff filtered, {R.get("identified_players",0)} players numbered.' if nim else "Fully offline run (no API)."
    css = ("body{margin:0;background:#0d1117;color:#e6edf3;font:14px/1.5 system-ui,Segoe UI,sans-serif}main{max-width:1100px;margin:auto;padding:24px 16px}h1{font-size:26px;margin:0}h2{font-size:16px;margin:28px 0 10px;border-bottom:1px solid #30363d;padding-bottom:6px}h2 em{font-weight:400;color:#8b949e;font-size:12px;margin-left:8px}"
           ".sub{color:#8b949e}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.tile{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px}.tile b{display:block;font-size:22px}.tile span{color:#8b949e;font-size:12px}"
           ".pitch{width:100%;border-radius:10px}ul{list-style:none;padding:0}li{background:#161b22;border-left:4px solid #8b949e;margin:8px 0;padding:10px 12px;border-radius:6px}li small{display:block;color:#8b949e}.p1{border-color:#ff5c5c}.p2{border-color:#ffa94d}"
           "table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:5px 6px;border-bottom:1px solid #21262d;text-align:left}td i{display:block;height:8px;background:#3fb950;border-radius:4px}pre{white-space:pre-wrap;background:#161b22;padding:14px;border-radius:10px;font:13px/1.55 system-ui}.warn{background:#2d2204;border:1px solid #6b5300;border-radius:10px;padding:12px;color:#f2cc60}.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:800px){.row{grid-template-columns:1fr}}")
    return (f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scouting report – {esc(club)}</title><style>{css}</style><main><h1>Opposition scouting report</h1><div class="sub">{esc(club)} · {R["duration"]} s analysed · kits <span style="color:{col(our)}">■ ours</span> <span style="color:{col(1-our)}">■ opponent</span></div>'
            f'<h2>Key numbers</h2><div class="tiles">{tiles}</div>{ai}<h2>Tactical findings <em>deterministic rules, evidence attached</em></h2><ul>{recs}</ul>'
            f'<div class="row"><section><h2>Average positions</h2>{pitch_svg(dots)}</section><section><h2>Opponent presence heat-map</h2>{pitch_svg(heat)}</section></div>'
            f'<h2>Player profiles <em>0–1 bars: speed · acceleration · press-resistance · coverage · work-rate</em></h2><table><tr><th>Player</th><th>Side</th><th>Role</th><th>Speed</th><th>Accel</th><th>Press-res.</th><th>Coverage</th><th>Work</th><th>Conf.</th></tr>{rows}</table>'
            f'<h2>Data quality &amp; limits</h2><div class="warn">{esc(R.get("scope",""))}<br>{prov}<br>Metrics use only players the broadcast camera shows. Thresholds are untuned defaults; the phase forecast is a per-clip heuristic and is not validated (see README, v4). Names appear only if the club supplied a team sheet matched by jersey number; no face recognition is used.</div></main>')
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("report"); ap.add_argument("out"); ap.add_argument("--club", default="Analysing club"); a = ap.parse_args()
    open(a.out, "w").write(build(json.load(open(a.report)), a.club)); print("wrote", a.out)
