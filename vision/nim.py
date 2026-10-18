"""Optional NVIDIA NIM (build.nvidia.com, OpenAI-compatible) assist. Off unless NVIDIA_API_KEY is set AND --nim is passed.
Key is read ONLY from the environment (never stored, never logged). Stdlib only (urllib) - no new dependency.
Uses are deliberately sparse and budgeted (hard cap on calls) so the real-time CPU pipeline stays the source of truth:
  * jersey_ocr(crop)  - VLM reads the shirt number (replaces weak Tesseract; the existing multi-frame vote still gates it)
  * classify(crop)    - once per track: player / goalkeeper / referee / other -> referees & staff are removed from team metrics
  * brief(report)     - text LLM turns the numeric report into a coach-ready briefing, constrained to the supplied numbers
Anything the API returns is validated; on any error/timeout/budget exhaustion we return None and the pipeline carries on offline."""
import os, json, base64, time, re, hashlib, urllib.request, urllib.error
import cv2, numpy as np

BASE = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
VLM = os.environ.get("NIM_VLM", "meta/llama-3.2-90b-vision-instruct")
LLM = os.environ.get("NIM_LLM", "meta/llama-3.1-70b-instruct")

class NimClient:
    def __init__(s, key=None, vlm=VLM, llm=LLM, budget=150, timeout=25, base=BASE):
        s.key = key or os.environ.get("NVIDIA_API_KEY", ""); s.vlm, s.llm, s.budget, s.timeout, s.base = vlm, llm, budget, timeout, base.rstrip("/")
        s.calls = s.errors = 0; s.cache = {}; s.dead = False; s.last = 0.0
    @property
    def ok(s): return bool(s.key) and not s.dead and s.calls < s.budget
    def _post(s, model, messages, max_tokens=200):
        if not s.ok: return None
        wait = 0.35 - (time.time() - s.last)                                        # <=~3 req/s: stay far below free-tier rate limits
        if wait > 0: time.sleep(wait)
        s.last = time.time(); s.calls += 1
        body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}).encode()
        for attempt in range(2):
            try:
                rq = urllib.request.Request(s.base + "/chat/completions", body, {"Authorization": "Bearer " + s.key, "Content-Type": "application/json", "Accept": "application/json"})
                with urllib.request.urlopen(rq, timeout=s.timeout) as r: return json.load(r)["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code in (401, 403): s.dead = True; print("NIM: key rejected (%d) - continuing offline" % e.code); return None
                if e.code == 429 and attempt == 0: time.sleep(2.0); continue
                s.errors += 1; return None
            except Exception: s.errors += 1; return None
            finally:
                if s.errors >= 8: s.dead = True                                        # circuit breaker: stop paying latency for a dead endpoint
        return None
    @staticmethod
    def _img(crop, side=224):
        h, w = crop.shape[:2]; k = side / max(h, w); crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90]); return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}}
    @staticmethod
    def _json(t):
        m = re.search(r"\{.*\}", t or "", re.S)
        try: return json.loads(m.group(0)) if m else None
        except Exception: return None
    def _vision(s, crop, prompt):
        if crop is None or crop.size == 0: return None
        key = hashlib.md5(crop.tobytes() + prompt.encode()).hexdigest()
        if key not in s.cache: s.cache[key] = s._json(s._post(s.vlm, [{"role": "user", "content": [{"type": "text", "text": prompt}, s._img(crop)]}]))
        return s.cache[key]
    def jersey_ocr(s, crop):
        """-> int 1..99 or None. The model is told to answer null unless the digits are clearly legible (we prefer silence to a wrong name)."""
        d = s._vision(crop, 'This is the back or front of a football player\'s shirt. Read the shirt NUMBER only if the digits are clearly legible. Reply with JSON only: {"number": <integer 1-99 or null>, "legible": <true|false>}. Do not guess.')
        try: n = d["number"]; return int(n) if d and d.get("legible") and n is not None and 1 <= int(n) <= 99 else None
        except Exception: return None
    def classify(s, crop):
        """-> 'player' | 'goalkeeper' | 'referee' | 'other' | None (unknown -> caller keeps the track)."""
        d = s._vision(crop, 'Football broadcast crop of one person on the pitch. Classify: "player" (outfield player), "goalkeeper", "referee" (match official, usually black/yellow/neon kit, or assistant referee), or "other" (coach, ball boy, photographer, steward). Reply with JSON only: {"role": "..."}')
        r = (d or {}).get("role"); return r if r in ("player", "goalkeeper", "referee", "other") else None
    def brief(s, facts, lang="English", club="the analysing club"):
        sys_ = ("You are a football analyst writing a concise pre-match scouting brief for a first-team coaching staff. Use ONLY the numbers and tags in FACTS. "
                "Never invent players, names, scores, or statistics. Every claim must cite the metric it comes from. If a figure is marked unreliable or coverage is low, say so plainly. "
                "Structure: 1) Opponent shape in one paragraph 2) Three exploitable patterns 3) Three risks/limits of this data 4) Suggested training-ground counter-measures. Write in " + lang + ". Max 350 words.")
        return s._post(s.llm, [{"role": "system", "content": sys_}, {"role": "user", "content": "Club: %s\nFACTS (JSON):\n%s" % (club, json.dumps(facts)[:9000])}], max_tokens=700)

def verify_brief(text, facts):
    """Fact-check: every number in the AI brief must appear (rounded) somewhere in FACTS. Returns {"unsupported": [...], "ok": bool}. Unsupported numbers are flagged, never silently kept."""
    flat = []
    def walk(o):
        if isinstance(o, dict): [walk(v) for v in o.values()]
        elif isinstance(o, (list, tuple)): [walk(v) for v in o]
        elif isinstance(o, (int, float)) and not isinstance(o, bool): flat.append(float(o))
        elif isinstance(o, str): flat.extend(float(m) for m in re.findall(r"\d+(?:\.\d+)?", o))
    walk(facts); bad = []
    for m in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text or ""):
        v = float(m)
        if v in (1, 2, 3, 4) and len(m) == 1: continue                                  # list numbering / "three patterns"
        if not any(abs(v - f) <= max(0.51, 0.02 * abs(f)) for f in flat): bad.append(m)
    return {"unsupported": sorted(set(bad)), "ok": not bad}
