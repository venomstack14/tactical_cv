"""Train + evaluate the 5-second-ahead team-phase model on real tracking data (Metrica Sports sample games 1-2, 25 Hz, anonymised).
   python -m tools.train_phase_model DATA_DIR      (DATA_DIR holds Sample_Game_{1,2}_RawTrackingData_{Home,Away}_Team.csv and ev{1,2}.csv)
Features come from a simulated BROADCAST VIEW (only players within +-30 m of the ball), the target from the full tracking truth."""
import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from analytics.tendency import PH, team_features, _adv, phase_of, L
STEP, H5, H3, VIEW = 5, 25, 15, 30.0          # 25 Hz -> 0.2 s grid; +5 s target; 3 s momentum; broadcast half-width (m)
def load(d, g):
    A = {}
    for s in ("Home", "Away"):
        df = pd.read_csv(f"{d}/Sample_Game_{g}_RawTrackingData_{s}_Team.csv", header=None, skiprows=3).iloc[::STEP]
        A[s] = df.iloc[:, 3:-2].values.reshape(len(df), -1, 2) * [L, 68.0]; ball = df.iloc[:, -2:].values * [L, 68.0]
    return df.iloc[:, 0].values, df.iloc[:, 2].values, A["Home"], A["Away"], ball
def build(d, g):
    per, t, home, away, ball = load(d, g); n = len(t); X, Y, cur, meta = [], [], [], []
    sg = lambda k, p: (1 if p == 1 else -1) * (1 if k == 0 else -1)           # home attacks +x in period 1; teams swap ends in period 2
    cfull = {0: np.full(n, np.nan), 1: np.full(n, np.nan)}; ccam = {0: np.full(n, np.nan), 1: np.full(n, np.nan)}; truth = {0: [None] * n, 1: [None] * n}
    adv = lambda arr, k, p: _adv(arr[:, 0], sg(k, p))
    for i in range(n):
        p = per[i]
        for k, arr in ((0, home), (1, away)):
            ok = ~np.isnan(arr[i, :, 0]); a = np.sort(adv(arr[i], k, p)[ok]); a = a[1:] if len(a) >= 5 else a
            if len(a) >= 3: cfull[k][i] = a.mean()
    for k in (0, 1):                                                          # truth phase from full tracking
        for i in range(n):
            if not np.isnan(cfull[k][i]): v = (cfull[k][i] - cfull[k][i - H3]) / 3 if i >= H3 and per[i - H3] == per[i] and not np.isnan(cfull[k][i - H3]) else 0.0; truth[k][i] = phase_of(cfull[k][i], v)
    feats = {}
    for i in range(n):
        if np.isnan(ball[i, 0]): continue
        for k in (0, 1):
            S, O = (home, away) if k == 0 else (away, home); p = per[i]
            ms = ~np.isnan(S[i, :, 0]) & (np.abs(S[i, :, 0] - ball[i, 0]) <= VIEW); mo = ~np.isnan(O[i, :, 0]) & (np.abs(O[i, :, 0] - ball[i, 0]) <= VIEW)
            if ms.sum() < 5 or mo.sum() < 5: continue
            c3 = ccam[k][i - H3] if i >= H3 and per[i - H3] == p else np.nan
            r = team_features(adv(S[i], k, p)[ms], S[i, ms, 1], adv(O[i], 1 - k, p)[mo], None if np.isnan(c3) else c3)
            if r is None: continue
            ccam[k][i] = r[2]; feats[(i, k)] = r
    for (i, k), (f, ph, c) in feats.items():
        j = i + H5
        if j < n and per[j] == per[i] and truth[k][j] is not None and i % 5 == 0:   # 1 s grid
            X.append(f); Y.append(PH.index(truth[k][j])); cur.append(PH.index(ph)); meta.append((g, k, t[i]))
    return np.array(X), np.array(Y), np.array(cur), meta
def evaluate(Xtr, Ytr, Xte, Yte, Cte):
    sc = StandardScaler().fit(Xtr); lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xtr), Ytr); gb = HistGradientBoostingClassifier(max_iter=150, learning_rate=.06).fit(Xtr, Ytr)
    P = {"majority": np.full(len(Yte), np.bincount(Ytr).argmax()), "persistence": Cte, "logreg": lr.predict(sc.transform(Xte)), "gbm": gb.predict(Xte)}; ch = Yte != Cte; out = {}
    for k, p in P.items(): out[k] = {"acc": round(float((p == Yte).mean()), 3), "macro_f1": round(float(f1_score(Yte, p, average="macro")), 3), "acc_when_phase_changes": round(float((p[ch] == Yte[ch]).mean()), 3)}
    prob = lr.predict_proba(sc.transform(Xte)); top2 = np.argsort(-prob, 1)[:, :2]; out["logreg"]["top2_when_changes"] = round(float(np.mean([Yte[i] in lr.classes_[top2[i]] for i in np.where(ch)[0]])), 3)
    out["n_test"], out["share_phase_changes"] = int(len(Yte)), round(float(ch.mean()), 3); return out
def possession_check(d, g):
    """Does our ATTACK phase coincide with that team actually having the ball (last event team)? Real events as ground truth."""
    ev = pd.read_csv(f"{d}/ev{g}.csv"); per, t, home, away, ball = load(d, g); ev = ev[ev["Type"].isin(["PASS", "SHOT", "RECOVERY", "BALL LOST", "CHALLENGE", "BALL OUT"])].sort_values("Start Time [s]")
    tt, tm = ev["Start Time [s]"].values, ev["Team"].values; res = {q: [0, 0] for q in PH}; cf = {0: [], 1: []}
    sg = lambda k, p: (1 if p == 1 else -1) * (1 if k == 0 else -1)
    for i in range(0, len(t), 5):
        j = np.searchsorted(tt, t[i]) - 1
        if j < 0: continue
        for k, arr, nm in ((0, home, "Home"), (1, away, "Away")):
            ok = ~np.isnan(arr[i, :, 0]); a = np.sort(_adv(arr[i, ok, 0], sg(k, per[i]))); a = a[1:] if len(a) >= 5 else a
            if len(a) < 3: continue
            ph = phase_of(a.mean(), 0.0 if i < H3 else 0.0); res["ATTACK" if a.mean() >= 55 else "DEFEND" if a.mean() <= 40 else "BUILD_UP"][0] += tm[j] == nm; res["ATTACK" if a.mean() >= 55 else "DEFEND" if a.mean() <= 40 else "BUILD_UP"][1] += 1
    return {q: round(v[0] / v[1], 3) for q, v in res.items() if v[1]}
if __name__ == "__main__":
    d = sys.argv[1]; D = {g: build(d, g) for g in (1, 2)}; print({g: (len(D[g][0])) for g in D}); rep = {}
    for tr, te in ((1, 2), (2, 1)): rep[f"train_game{tr}_test_game{te}"] = evaluate(D[tr][0], D[tr][1], D[te][0], D[te][1], D[te][2])
    rep["possession_vs_events"] = {f"game{g}": possession_check(d, g) for g in (1, 2)}
    X = np.vstack([D[g][0] for g in D]); Y = np.concatenate([D[g][1] for g in D]); sc = StandardScaler().fit(X); lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X), Y)
    json.dump({"classes": [PH[c] for c in lr.classes_], "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(), "horizon_s": 5,
               "trained_on": "Metrica Sports sample games 1-2 (real tracking, broadcast-view simulated)", "features": ["centroid_depth", "momentum_3s", "width", "depth_spread", "centroid_overlap", *["phase_" + q for q in PH]]}, open("models/phase_forecast.json", "w"))
    json.dump(rep, open("models/phase_forecast_eval.json", "w"), indent=1); print(json.dumps(rep, indent=1))
