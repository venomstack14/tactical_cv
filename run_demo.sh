#!/bin/sh
set -e
python -m tools.make_demo_video demo.mp4 --seconds 8
python main.py demo.mp4 --calib demo_calib.json --our-color red --no-show --out out.mp4 --report report.json
python -m tools.accuracy_synth report.json demo_truth.json     # measured on SYNTHETIC truth only
python -m tools.make_report report.json scouting.html --club "Demo FC"
python -m tools.make_reel report.json reel.mp4                 # 1080x1920 vertical clip for socials
python -m tools.make_clips out.mp4 report.json clips/          # one clip per key tactical moment
