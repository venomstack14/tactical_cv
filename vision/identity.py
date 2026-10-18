"""Jersey-number reading (OCR, optional pytesseract) + club-supplied team sheet -> names. NO face recognition.
Numbers are voted over many frames; broadcast shots are often too small/oblique, so most tracks stay unidentified (that is reported, not hidden)."""
import json, cv2, numpy as np
from collections import defaultdict
try: import pytesseract
except Exception: pytesseract = None

def ocr_number(img):
    """Return int 1..99 or None. Two Tesseract page-modes must agree (cuts random misreads; stock OCR is still weak - see README)."""
    if pytesseract is None or img is None or img.size == 0: return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC); g = cv2.createCLAHE(2.0, (4, 4)).apply(g)
    for inv in (False, True):
        _, b = cv2.threshold(g, 0, 255, (cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY) + cv2.THRESH_OTSU)
        b = cv2.copyMakeBorder(b, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=int(b[0, 0])); rd = []
        for psm in (8, 13):
            t = "".join(c for c in pytesseract.image_to_string(b, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789") if c.isdigit()); rd.append(t)
        if rd[0] and rd[0] == rd[1] and 1 <= int(rd[0]) <= 99 and len(rd[0]) <= 2: return int(rd[0])
    return None

class TeamSheet:
    """JSON the CLUB provides: {"ours": {"7": "Name"}, "opp": {"9": "Name"}}. Names come from this file only."""
    def __init__(s, path=None): s.d = json.load(open(path)) if path else {}
    def name(s, side, number): return s.d.get(side, {}).get(str(number))

class JerseyReader:
    def __init__(s, min_h=70, every=15, min_votes=3, min_share=0.6, ocr=ocr_number):
        s.min_h, s.every, s.min_votes, s.min_share, s.ocr = min_h, every, min_votes, min_share, ocr
        s.votes = defaultdict(lambda: defaultdict(int)); s.tries = defaultdict(int)
    def observe(s, frame, tid, box, frame_idx):
        x1, y1, x2, y2 = [int(v) for v in box]; h, w = y2 - y1, x2 - x1
        if h < s.min_h or (frame_idx + tid) % s.every: return
        crop = frame[max(y1 + int(.15 * h), 0):y1 + int(.55 * h), max(x1 + int(.2 * w), 0):x2 - int(.2 * w)]
        s.tries[tid] += 1; n = s.ocr(crop)
        if n is not None: s.votes[tid][n] += 1
    def number(s, tid):
        v = s.votes.get(tid)
        if not v: return None
        n, c = max(v.items(), key=lambda kv: kv[1])
        return n if c >= s.min_votes and c / sum(v.values()) >= s.min_share else None
    def identities(s, team_of, our, sheet=None):
        """-> {tid: {"number", "name", "team"}} ; if two tracks of one team claim a number, the better-voted one keeps it."""
        best = {}
        for tid in s.votes:
            n = s.number(tid)
            if n is None: continue
            k = (team_of.get(tid, -1), n)
            if k not in best or s.votes[tid][n] > s.votes[best[k]][n]: best[k] = tid
        out = {}
        for (tm, n), tid in best.items():
            side = "ours" if tm == our else "opp"
            out[tid] = {"number": n, "team": tm, "name": sheet.name(side, n) if sheet else None}
        return out
