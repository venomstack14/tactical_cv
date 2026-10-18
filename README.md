<div align="center">

# ⚽ tactical_cv

**Football tactical analysis from raw broadcast video — on a laptop CPU.**
Pitch calibration · player tracking · team shape · pressing · auto key-moment clips · coach-ready report · optional NVIDIA NIM assist

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![cpu](https://img.shields.io/badge/runs%20on-CPU%20only-brightgreen) ![license](https://img.shields.io/badge/license-MIT-lightgrey) ![nvidia](https://img.shields.io/badge/NVIDIA%20NIM-optional-76b900)

<img src="assets/demo.gif" width="300" alt="vertical tactical replay"/>

</div>

## One command
```bash
pip install -r requirements.txt && sh run_demo.sh      # or: docker build -t tcv . && docker run --rm -v $PWD/out:/app/out tcv
```
Produces `report.json`, `scouting.html` (club-style page), `reel.mp4` (1080x1920 social clip), `clips/` (one per key moment).

## Measured accuracy (honest)
| test | result |
|---|---|
| Synthetic clip, exact ground truth (`tools.accuracy_synth`) | mean position error **0.38 m**, p90 0.71 m, team-label accuracy 97.9%, detection recall 89.7% |
| Real broadcast footage | **not yet measured** - see `tools.validate`; contributions of labelled clips welcome |
Synthetic numbers show the geometry/tracking maths is right; they say nothing about detector quality on real matches.

## Roadmap (PRs welcome)
- [ ] Automatic pitch-keypoint calibration (removes the manual 4-landmark step)
- [ ] Football-trained detector + re-identification for occlusions
- [ ] Labelled real clips + published accuracy table
- [ ] True PPDA / line-break passes from event data

# tactical_cv - offline CPU football tactical analysis

Standalone: no API calls, no GPU, no torch at runtime. Needs `numpy scipy opencv-python`.

## Quick start
```bash
pip install -r requirements.txt
python -m tools.make_demo_video demo.mp4 --seconds 8          # synthetic clip + demo_calib.json (known truth)
python main.py demo.mp4 --calib demo_calib.json --our-color red --out out.mp4 --report report.json   # add --no-show on headless machines
python -m tests.test_all
```
Real footage: pick >= 4 non-collinear pitch landmarks on the first frame (`--interactive`, or a JSON like `demo_calib.json`
with names from `core/pitch.py:LANDMARKS`; pixel coords are in the 960-wide processing frame).
For YOLOv8n: run `python -m tools.export_onnx` once (online), then `--weights yolov8n.onnx --detect-every 3`.
Pitch frame: x = length left->right as the main camera sees it, y = from the near touchline (0) to the far one (68).
`--attack right|left` = direction OUR team attacks in that frame; `--our-color` picks our kit cluster.

## Layout
| file | role |
|---|---|
| `vision/calibration.py` | HSV pitch + white-line masks, Hough segments/intersections, RANSAC homography init, LK propagation, ECC line refinement, keypoint (YOLO-pose) solver with gating |
| `vision/tracker.py` | YOLOv8n-ONNX / ultralytics / blob detectors, own ByteTrack, torso-colour team classifier, zero-shot `PlayerProfile` vector |
| `engine/tactics.py` | 5 s rolling-window metrics + deterministic Juego de Posicion rule engine with hysteresis and evidence strings |
| `vision/render.py`, `main.py` | split view (video + warnings | radar + panel), CLI, JSON report |

## Profile vector (all 0..1, built only from what the camera saw)
speed_index (sustained 1 s speed / 9 m/s) | accel_index (p95 positive accel / 5 m/s2) | press_resistance (speed kept when an opponent is < 4 m; 0.5 = unknown) |
pitch_coverage (5 m cells visited / 40) | work_rate (mean speed / 3 m/s) + role from the mean offset to the team centroid + confidence (observed seconds / 20).

## Known limits (read before trusting numbers)
* Metrics use only players the broadcast camera shows; below 7 visible opponents the engine says "low coverage" instead of guessing.
* Ball is only used if the detector returns one; pressing is inferred from opponents closing on our players.
* Thresholds in `engine/tactics.py:Thresholds` are sensible defaults, NOT tuned on real matches - calibrate them with your analysts.
* Role/press_resistance are proxies; the profile needs ~10-20 s of observation per player to be meaningful.
* Line-ECC refinement can fail on cluttered footage (ads, graphics): it is gated, and the system then holds the last homography (`calib HOLD`).

## v2 additions
| feature | how | honest status |
|---|---|---|
| Team physical totals (distance, top speed, sprints) | `--report` -> `team_stats` (players observed >= 5 s) | Sums only players the camera showed; not full-team totals |
| Possession / passes / turnovers | `analytics/matchstats.py`, uses detected ball | Proxy. Marked `reliable: false` if ball seen < 30% of frames or < 20 s controlled. Ball is projected to the ground plane |
| Jersey numbers | `--jersey` (pytesseract + Tesseract binary), `vision/identity.py` | Stock OCR is weak (misreads two-digit numbers even on clean synthetic digits); needs tall player boxes (>= 70 px). Votes over frames, two OCR passes must agree. Swap in a trained digit model for real use |
| Player names | `--team-sheet sheet.json` `{"ours":{"7":"Name"},"opp":{...}}` | Names come ONLY from the club's own sheet, matched by jersey number. No face recognition by design |
| Validation | `python -m tools.validate report.json labels.json` | Compares engine tags to analyst labels (P/R/F1, flags tags needing retuning). Needs real labelled footage - not supplied |
| Setup | still manual calibration + attack direction + kit colour | Unchanged; cluttered footage can still break calibration |
Report also contains a `timeline` of tags per 0.5 s for validation.

## v3: tendencies and forecast (`analytics/tendency.py`)
* Per player: attack_score 0..1 (mean depth vs own team centroid, forward vs recovery runs, final/own-third time) -> Attacking / Balanced / Defensive, with confidence.
* Per team, every second: DEFEND / DEFENSIVE_TRANSITION / BUILD_UP / ATTACKING_TRANSITION / ATTACK from centroid depth and momentum.
* Forecast: the opponent's observed phase-to-phase habits so far in the clip (+ favoured attacking channel). Heuristic, low confidence on short clips; NOT a trained predictor.
* No face recognition (deliberate). Identify players via jersey numbers + a club-supplied team sheet.

## v4: training on real data (negative result, kept on purpose)
`python -m tools.train_phase_model DATA_DIR` trains/evaluates a 5-second-ahead team-phase model on **Metrica Sports sample games 1-2** (real, anonymised 25 Hz tracking + events; github.com/metrica-sports/sample-data). Features come from a simulated broadcast view (players within 30 m of the ball); train on one game, test on the other.
Results (`models/phase_forecast_eval.json`): persistence ("phase stays the same") 0.68-0.71 accuracy; logistic regression 0.58-0.64; gradient boosting 0.62-0.64. When the phase actually changes, models are right only ~19-27% (top-2 ~49-53%). **No model beat the naive baseline, so none is shipped.** The page's forecast is the per-clip habit heuristic and is labelled unvalidated.
Phase vs real possession (event data): ATTACK phase coincided with that team having the ball 57% (game 1) and 43% (game 2) of the time. Phases describe team shape/territory, not possession.
Not validated at all: attack_score tendencies, jersey OCR on real footage, possession proxy on real broadcast video, detector accuracy.

## v5: NVIDIA NIM assist (optional) + club-ready report
```bash
export NVIDIA_API_KEY=nvapi-...            # from build.nvidia.com; read from the environment only, never stored or logged
python main.py match.mp4 --calib c.json --our-color white --nim --nim-budget 150 --club "Your FC" --lang English --report report.json --out out.mp4 --no-show
python -m tools.make_report report.json scouting.html --club "Your FC"      # self-contained page to present
```
What `--nim` adds (every call is budgeted, cached, rate-limited; any failure falls back to the offline result, circuit-breaker after 8 errors):
* **VLM jersey-number OCR** instead of Tesseract (still gated by the multi-frame vote, so a wrong read needs to repeat to count). Names still come only from the club team sheet.
* **Referee / staff filter**: one VLM check per track removes officials from the team metrics (they pollute kit clustering and block shape).
* **AI scouting brief** (text LLM, temperature 0) written only from the report's numbers, with limits stated; stored in `report.json["ai_brief"]`.
Models are configurable: `NIM_VLM`, `NIM_LLM`, `NVIDIA_BASE_URL` (defaults: llama-3.2-90b-vision-instruct, llama-3.1-70b-instruct). Check they are still listed on build.nvidia.com.
Honest status: the client is tested against a local mock server only (this dev box could not reach NVIDIA); accuracy gains are NOT measured - run `tools.validate` on labelled footage before claiming numbers to a club. Per-frame detection stays local (API latency makes it unsuitable); use YOLO weights for real footage.

## v8: real edited broadcast footage (`tools/analyze_raw.py`, `tools/make_reel_real.py`)
```bash
python -m tools.analyze_raw clip.mp4 annot.mp4 stats.json --weights yolov8m.pt --imgsz 1280 --stride 3   # weights: github.com/ultralytics/assets/releases
python -m tools.make_reel_real annot.mp4 stats.json reel.mp4 --start 11.6 --end 28.4 --repo github.com/you/tactical_cv
```
Shot-cut detection -> wide-shot typing -> YOLOv8m person/ball detection (every 3rd frame, CPU) -> ByteTrack with **camera-motion compensation** (LK + RANSAC similarity warp of track states) -> kit-colour teams (2-means over wide shots; yellow officials excluded per track).
Tested on a 28 s Real Madrid-PSG highlights recording: 16.6 s tracked wide shots, 12 max simultaneous players. Image-space only: no metres/speeds/formations on a panning, edited feed.
**Known weaknesses (measured):** ~2-3 track IDs per real player (`id_fragmentation` in stats.json) because there is no appearance re-ID; players overlapping share/lose boxes; ball visibility is sampled every 3rd frame.

## v9: ID-switch repair (`vision/reid.py`, `tools/eval_reid.py`)
Offline tracklet stitching per shot: camera-compensated reference positions + HSV appearance descriptor, speed-gated, same-team, one-to-one, no temporal overlap (Hungarian). Same-kit players look alike, so motion continuity does most of the work.
Real clip: shot 1 IDs 36 -> 23 (3.0 -> 1.9 IDs per max-simultaneous player), shot 2 unchanged (14, ~1.75). Linker test on real tracklets with synthetic breaks (`python -m tools.eval_reid x.tracklets.pkl`, 60 trials): 85% rejoined correctly, 13% wrong link, 2% left apart. Small sample; some "wrong" links may be genuine continuations of other fragments of the same player.
Remaining ID churn is partly legitimate (players leaving and re-entering a panning view) and needs a learned re-ID embedding + labelled data to go further.
