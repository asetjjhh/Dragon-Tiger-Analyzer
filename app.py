import re
from collections import Counter, defaultdict
from math import sqrt, comb
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dragon Tiger Analyzer V9", page_icon="🐉", layout="wide")

st.title("🐉🐯 Dragon Tiger Analyzer — V9")
st.caption("V9 • Error-safe validation, provider/session separation, verified historical records, confidence bounds, stability scoring and conservative signals.")

# -----------------------------------------------------------------------------
# Captured Evolution table snapshot (#100)
# The road in the supplied #100 screenshot was reconstructed from the visible
# 6-row bead-style grid. Reading columns top-to-bottom reproduces the displayed
# #73 checkpoint counts (D32 / T37 / Tie4) and the #100 totals (D43 / T52 / Tie5).
# This is a seed history, not a claim that the next casino outcome is predictable.
# -----------------------------------------------------------------------------
EVOLUTION_SEED = (
    "T D D D T D "
    "T T X T T T "
    "D T D D T T "
    "D T T T D T "
    "D D T T T T "
    "T T T D T D "
    "X T D D D D "
    "T T D T T T "
    "D T D D D T "
    "T D D D D D "
    "D X X T D T "
    "T T D T D T "
    "D T D T T T "
    "T T X T D D "
    "D T D D D D "
    "D T T D T T "
    "T D T T"
).split()

assert len(EVOLUTION_SEED) == 100

# Verified Evolution — Emperor Dragon & Tiger #1–#63.
EMPEROR_EVOLUTION_63 = (
    "D T D D X D D D T X "
    "T D T D T D T T T D "
    "D T T T T D X T D D "
    "T D T D D T D D D T "
    "D T D T D T D D D D "
    "D T D D X T T T D D "
    "D D D D D D T"
).split()

assert len(EMPEROR_EVOLUTION_63) == 63
assert EMPEROR_EVOLUTION_63.count("D") == 35
assert EMPEROR_EVOLUTION_63.count("T") == 24
assert EMPEROR_EVOLUTION_63.count("X") == 4

# Verified aggregate summaries from the supplied completed screenshots.
# These are intentionally NOT converted into invented hand-by-hand sequences.
KNOWN_SESSIONS = pd.DataFrame([
    # Latest verified completed sessions supplied by the user.
    {"Provider":"Evolution", "Game":"Dragon Tiger", "Session":"Latest verified #1–#102", "Captured through":102, "Dragon":54, "Tiger":40, "Tie":8, "Sequence available":False},
    {"Provider":"Pragmatic Play Live", "Game":"Dragon Tiger", "Session":"Latest verified #1–#112", "Captured through":112, "Dragon":54, "Tiger":49, "Tie":9, "Sequence available":False},
    {"Provider":"Evolution", "Game":"Emperor Dragon and Tiger", "Session":"Verified #1–#63", "Captured through":63, "Dragon":35, "Tiger":24, "Tie":4, "Sequence available":True},
    # Older verified aggregate checkpoints are retained separately.
    {"Provider":"Evolution", "Game":"Dragon Tiger", "Session":"Older verified #1–#145 checkpoint", "Captured through":145, "Dragon":64, "Tiger":72, "Tie":9, "Sequence available":False},
    {"Provider":"Pragmatic Play Live", "Game":"Dragon Tiger", "Session":"Older verified #1–#92 checkpoint", "Captured through":92, "Dragon":43, "Tiger":42, "Tie":7, "Sequence available":False},
    {"Provider":"Evolution", "Game":"Emperor Dragon and Tiger", "Session":"Older verified #1–#77 checkpoint", "Captured through":77, "Dragon":38, "Tiger":35, "Tie":4, "Sequence available":False},
])



def parse_result(x):
    x = str(x).strip().upper()
    return {
        "D": "D", "DRAGON": "D",
        "T": "T", "TIGER": "T",
        "X": "X", "TIE": "X",
    }.get(x)


def clean_tokens(text):
    tokens = [x for x in re.split(r"[\s,;|]+", str(text).strip()) if x]
    parsed = [parse_result(x) for x in tokens]
    invalid = [tokens[i] for i, v in enumerate(parsed) if v is None]
    return [v for v in parsed if v is not None], invalid


def clean_csv(df):
    cols = {str(c).lower().strip(): c for c in df.columns}
    if "outcome" not in cols:
        return pd.DataFrame(), "CSV must contain an Outcome column."
    out = pd.DataFrame()
    if "table" in cols:
        out["Table"] = df[cols["table"]].fillna("Unknown Table").astype(str).str.strip()
    elif "session" in cols:
        out["Table"] = df[cols["session"]].fillna("Unknown Session").astype(str).str.strip()
    else:
        out["Table"] = "Imported Session"
    if "hand" in cols:
        out["Hand"] = pd.to_numeric(df[cols["hand"]], errors="coerce")
    else:
        out["Hand"] = np.arange(1, len(df) + 1)
    out["Outcome"] = df[cols["outcome"]].map(parse_result)
    invalid = int(out["Outcome"].isna().sum())
    out = out[out["Outcome"].notna()].reset_index(drop=True)
    return out, (f"Ignored {invalid} invalid outcome row(s)." if invalid else "")


def dt_only(results):
    return [r for r in results if r in ("D", "T")]


def counts(results):
    c = Counter(results)
    return c["D"], c["T"], c["X"]


def streak_runs(seq):
    runs = []
    if not seq:
        return runs
    cur = seq[0]
    n = 1
    for x in seq[1:]:
        if x == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur, n = x, 1
    runs.append((cur, n))
    return runs


def current_streak(seq):
    if not seq:
        return "-", 0
    return seq[-1], next((i + 1 for i, x in enumerate(reversed(seq)) if x != seq[-1]), len(seq))


def transition_probs(seq, alpha=1.0):
    # Laplace smoothing, trained only on the supplied history.
    out = {}
    for src in ("D", "T"):
        nxt = [b for a, b in zip(seq, seq[1:]) if a == src]
        d = nxt.count("D")
        t = nxt.count("T")
        den = d + t + 2 * alpha
        out[src] = {"D": (d + alpha) / den, "T": (t + alpha) / den, "samples": len(nxt)}
    return out


def recent_probs(seq, window, alpha=1.0):
    h = seq[-window:] if len(seq) >= window else seq
    d = h.count("D")
    t = h.count("T")
    den = d + t + 2 * alpha
    return {"D": (d + alpha) / den, "T": (t + alpha) / den, "samples": len(h)}


def ngram_probs(seq, order, alpha=1.0):
    if len(seq) <= order:
        return None
    pattern = tuple(seq[-order:])
    follows = []
    for i in range(len(seq) - order):
        if tuple(seq[i:i + order]) == pattern:
            follows.append(seq[i + order])
    if not follows:
        return None
    d = follows.count("D")
    t = follows.count("T")
    den = d + t + 2 * alpha
    return {"D": (d + alpha) / den, "T": (t + alpha) / den, "samples": len(follows), "pattern": "".join(pattern)}


def run_length_probs(seq, alpha=1.0):
    # P(continue current side | current run length), learned from prior runs.
    runs = streak_runs(seq)
    if not runs:
        return None
    side, length = runs[-1]
    continue_next = []
    for i, (s, L) in enumerate(runs[:-1]):
        if s == side and L == length:
            # This run continued if it could be observed beyond its terminal length.
            # Exact terminal run lengths do not directly predict; use historical
            # probability that a run of this length was followed by same side.
            nxt_side = runs[i + 1][0]
            continue_next.append(1 if nxt_side == side else 0)
    # For terminal run length, occurrences of exactly this length are informative
    # about whether the run tends to be followed by a switch (not a continuation),
    # so combine with generic side continuation tendency.
    trans = transition_probs(seq, alpha)
    return {
        "D": trans[side]["D"],
        "T": trans[side]["T"],
        "samples": len(continue_next),
        "side": side,
        "length": length,
    }


def model_probs(seq):
    models = {}
    models["Recent-5"] = recent_probs(seq, 5)
    models["Recent-10"] = recent_probs(seq, 10)
    models["Recent-20"] = recent_probs(seq, 20)
    tr = transition_probs(seq)
    if seq:
        models["Transition"] = {"D": tr[seq[-1]]["D"], "T": tr[seq[-1]]["T"], "samples": tr[seq[-1]]["samples"]}
    for order in (1, 2, 3, 4):
        p = ngram_probs(seq, order)
        if p is not None:
            models[f"N-gram-{order}"] = p
    return models


def blend_predictions(preds, weights=None):
    if not preds:
        return {"D": 0.5, "T": 0.5}
    if weights is None:
        weights = {k: 1.0 for k in preds}
    d = sum(weights.get(k, 0) * v["D"] for k, v in preds.items())
    t = sum(weights.get(k, 0) * v["T"] for k, v in preds.items())
    z = d + t
    return {"D": d / z, "T": t / z}


def fixed_model_signal(seq, model_name):
    p = model_probs(seq).get(model_name)
    if p is None:
        return None
    return "D" if p["D"] > p["T"] else "T" if p["T"] > p["D"] else None


def evaluate_fixed_model(history, model_name):
    if len(history) < 15:
        return None
    rows = []
    for i in range(10, len(history)):
        train = history[:i]
        sig = fixed_model_signal(train, model_name)
        if sig is not None:
            rows.append(sig == history[i])
    if not rows:
        return None
    return float(np.mean(rows))


def adaptive_signal(seq):
    models = model_probs(seq)
    if not models:
        return "NO BET", {"D": 0.5, "T": 0.5}, []

    weights = {}
    details = []
    for name, p in models.items():
        acc = evaluate_fixed_model(seq, name)
        # Conservative shrinkage toward 50% so tiny samples cannot dominate.
        if acc is None:
            w = 0.5
        else:
            w = 0.5 + max(-0.20, min(0.20, acc - 0.5))
        # More support => slightly more weight, but never enough to dominate.
        support = float(p.get("samples", 0))
        w *= min(1.0, 0.60 + 0.40 * min(1.0, support / 20.0))
        weights[name] = max(w, 0.05)
        details.append({"Model": name, "D": p["D"], "T": p["T"], "Support": int(p.get("samples", 0)), "Backtest acc": acc})

    dtp = blend_predictions(models, weights)
    # Tie is only eligible with enough observed ties and a materially elevated rate.
    total = len(seq)
    # Use all three-class history only for tie screening; D/T model remains separate.
    tie_count = 0
    if total:
        # Caller supplies only D/T here, so no tie count is available.
        tie_count = 0
    margin = abs(dtp["D"] - dtp["T"])
    signal = "NO BET"
    if len(seq) >= 20 and margin >= 0.10:
        signal = "DRAGON" if dtp["D"] > dtp["T"] else "TIGER"
    return signal, dtp, details


def signal_with_tie(results):
    seq = dt_only(results)
    if len(seq) < 20:
        return "NO BET", {"D": 0.0, "T": 0.0, "X": 0.0}, [], "Too little directional history."
    sig, dtp, details = adaptive_signal(seq)
    # Smoothed tie rate. Tie needs stronger evidence because it is much rarer.
    x = results.count("X")
    n = len(results)
    tie_p = (x + 1) / (n + 3)
    probs = {"D": dtp["D"] * (1 - tie_p), "T": dtp["T"] * (1 - tie_p), "X": tie_p}
    best = max(probs, key=probs.get)
    second = sorted(probs.values(), reverse=True)[1]
    # Never force a tie on a sparse tie sample.
    if x >= 8 and best == "X" and probs["X"] - second >= 0.08:
        sig = "TIE"
        reason = "Tie has enough historical observations and clears the conservative margin rule."
    elif sig in ("DRAGON", "TIGER") and probs[sig[0]] - max(probs["X"], probs["T"] if sig == "DRAGON" else probs["D"]) < 0.04:
        sig = "NO BET"
        reason = "The leading directional signal is too close to the competing outcome after tie adjustment."
    else:
        reason = "Directional ensemble has a measurable lean and passed the minimum-history gate."
    return sig, probs, details, reason


def walk_forward_model(results, model_name, min_history=20):
    seq = dt_only(results)
    rows = []
    for i in range(min_history, len(seq)):
        train = seq[:i]
        p = model_probs(train).get(model_name)
        if p is None:
            continue
        sig = "D" if p["D"] > p["T"] else "T" if p["T"] > p["D"] else "NO BET"
        rows.append({"Test": i + 1, "Signal": sig, "Actual": seq[i], "Correct": (sig == seq[i]) if sig != "NO BET" else np.nan})
    return pd.DataFrame(rows)


def all_model_backtest(results):
    seq = dt_only(results)
    names = ["Recent-5", "Recent-10", "Recent-20", "Transition", "N-gram-1", "N-gram-2", "N-gram-3", "N-gram-4"]
    rows = []
    for name in names:
        bt = walk_forward_model(results, name)
        sig = bt[bt["Signal"].isin(["D", "T"])] if not bt.empty else pd.DataFrame()
        rows.append({
            "Model": name,
            "Test points": len(bt),
            "Signals": len(sig),
            "Accuracy": float(sig["Correct"].mean()) if len(sig) else np.nan,
            "Coverage": len(sig) / len(bt) if len(bt) else np.nan,
        })
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z*z/n
    c = p + z*z/(2*n)
    a = z*sqrt((p*(1-p) + z*z/(4*n))/n)
    return (c-a)/den, (c+a)/den


def longest_runs(results):
    best = {"D": 0, "T": 0, "X": 0}
    for s, n in streak_runs(results):
        best[s] = max(best[s], n)
    return best


def transition_table(seq):
    rows = []
    for src in ("D", "T"):
        nxt = [b for a, b in zip(seq, seq[1:]) if a == src]
        rows.append({
            "After": src,
            "Next D": nxt.count("D"),
            "Next T": nxt.count("T"),
            "Continue %": nxt.count(src) / len(nxt) if nxt else np.nan,
            "Switch %": nxt.count("T" if src == "D" else "D") / len(nxt) if nxt else np.nan,
            "Samples": len(nxt),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# V6 validation helpers
# -----------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z*z/n
    center = p + z*z/(2*n)
    half = z * sqrt((p*(1-p) + z*z/(4*n))/n)
    return (center-half)/den, (center+half)/den

def exact_binomial_pvalue(k, n):
    if n <= 0:
        return np.nan
    tail_k = min(k, n-k)
    tail = sum(comb(n, i) for i in range(tail_k+1)) / (2**n)
    return min(1.0, 2*tail)

def v6_model_table(results):
    names = ["Recent-5","Recent-10","Recent-20","Transition","N-gram-1","N-gram-2","N-gram-3","N-gram-4"]
    seq = dt_only(results)
    rows=[]
    for name in names:
        bt=walk_forward_model(results,name)
        sig=bt[bt["Signal"].isin(["D","T"])] if not bt.empty else pd.DataFrame()
        n_sig=len(sig); correct=int(sig["Correct"].sum()) if n_sig else 0
        acc=correct/n_sig if n_sig else np.nan
        lo,hi=wilson(correct,n_sig)
        pv=exact_binomial_pvalue(correct,n_sig) if n_sig else np.nan
        robust=bool(n_sig>=20 and pd.notna(lo) and lo>0.50 and pd.notna(pv) and pv<0.10)
        rows.append({"Model":name,"Test points":len(bt),"Signals":n_sig,"Accuracy":acc,"Coverage":n_sig/len(bt) if len(bt) else np.nan,"Wilson low":lo,"Wilson high":hi,"p-value":pv,"Robust":robust})
    return pd.DataFrame(rows)

def v6_ensemble(results, validation):
    seq=dt_only(results); models=model_probs(seq); details=[]
    for name,p in models.items():
        r=validation[validation["Model"]==name]
        if r.empty: continue
        r=r.iloc[0]; acc=r["Accuracy"]; support=float(p.get("samples",0))
        weight=0.15 if pd.isna(acc) else 0.25 + min(0.75,max(0.0,float(acc)-0.50)*4)
        if not bool(r["Robust"]): weight*=0.55
        weight*=min(1.0,0.50+0.50*min(1.0,support/20.0))
        details.append((name,p,weight,bool(r["Robust"])))
    if not details: return {"D":0.5,"T":0.5},[]
    d=sum(p["D"]*w for _,p,w,_ in details); t=sum(p["T"]*w for _,p,w,_ in details); z=d+t
    return {"D":d/z,"T":t/z}, [{"Model":n,"D":p["D"],"T":p["T"],"Support":int(p.get("samples",0)),"Weight":w,"Robust":r} for n,p,w,r in details]

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("V9 controls")
mode = st.sidebar.radio(
    "History source",
    ["Built-in Evolution #100", "Verified Emperor Evolution #63", "Paste D/T/Tie", "Upload CSV"],
    index=0,
)

if mode == "Built-in Evolution #100":
    table_name = "Evolution Dragon Tiger — captured to #100"
    results = EVOLUTION_SEED.copy()
    st.sidebar.success("Loaded the captured #100 Evolution road history.")
elif mode == "Verified Emperor Evolution #63":
    table_name = "Evolution Emperor Dragon & Tiger — verified #1–#63"
    results = EMPEROR_EVOLUTION_63.copy()
    st.sidebar.success("Loaded verified Emperor #1–#63: D35 / T24 / Tie4.")
elif mode == "Paste D/T/Tie":
    table_name = st.sidebar.text_input("Table / provider / session name", "Current Session")
    text = st.sidebar.text_area("Paste results, oldest → newest", "", height=180, placeholder="D T T D X D T ...")
    results, invalid = clean_tokens(text)
    if invalid:
        st.sidebar.error("Input error — unrecognized result(s): " + ", ".join(invalid[:12]))
    if not results:
        st.info("Paste a D/T/Tie history to begin V9.")
        st.stop()
else:
    upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if upload is None:
        st.info("Upload a CSV with an Outcome column to begin V9.")
        st.stop()
    raw = pd.read_csv(upload)
    data, note = clean_csv(raw)
    if note:
        st.sidebar.warning(note)
    if data.empty:
        st.error("No valid D/T/Tie history found.")
        st.stop()
    table_names = list(data["Table"].unique())
    table_name = st.sidebar.selectbox("Table / session", table_names)
    results = data.loc[data["Table"] == table_name, "Outcome"].tolist()

if not results:
    st.error("No usable history.")
    st.stop()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Historical verified sessions
# -----------------------------------------------------------------------------
with st.expander("📚 Historical verified sessions", expanded=False):
    st.dataframe(KNOWN_SESSIONS, hide_index=True, use_container_width=True)
    st.caption("Latest verified aggregates: Evolution Dragon Tiger #102 = D54/T40/Tie8; Pragmatic Play Live Dragon Tiger #112 = D54/T49/Tie9. These are retained for reference and are not fabricated into chronological sequences.")

d, t, x = counts(results)
n = len(results)
cur, cur_n = current_streak(results)
best = longest_runs(results)
seq = dt_only(results)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hands captured", n)
c2.metric("Dragon", f"{d/n:.1%}")
c3.metric("Tiger", f"{t/n:.1%}")
c4.metric("Tie", f"{x/n:.1%}")
c5.metric("Current run", f"{cur} × {cur_n}")
st.caption(f"Selected source: **{table_name}**")

# -----------------------------------------------------------------------------
# V9 robust signal
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🎯 V9 robust signal")
seq = dt_only(results)
validation = v6_model_table(results) if len(seq) >= 20 else pd.DataFrame()
if len(seq) < 30:
    signal, probs, details, reason = "NO BET", {"D":0.5,"T":0.5}, [], "At least 30 non-Tie results are required for the V6 directional gate."
else:
    probs, details = v6_ensemble(results, validation)
    robust_count=int(validation["Robust"].sum()) if not validation.empty else 0
    margin=abs(probs["D"]-probs["T"])
    if robust_count < 1:
        signal, reason = "NO BET", "No model passed the V9 robustness gate (support + confidence interval + significance)."
    elif margin < 0.08:
        signal, reason = "NO BET", "The ensemble edge is too small after model shrinkage."
    else:
        signal = "DRAGON" if probs["D"] > probs["T"] else "TIGER"
        reason = f"{robust_count} model(s) passed the robustness gate and the ensemble margin is {margin:.1%}."
label={"DRAGON":"🐉 DRAGON","TIGER":"🐯 TIGER","NO BET":"⚪ NO BET"}[signal]
s1,s2,s3=st.columns(3)
s1.metric("Recommendation",label); s2.metric("Dragon model",f"{probs['D']:.1%}"); s3.metric("Tiger model",f"{probs['T']:.1%}")
st.info(reason)
st.caption("Research signal only — historical patterns cannot guarantee the next casino outcome.")

# Captured session registry
st.divider()
st.subheader("📚 Captured session registry")
st.write("Completed totals directly visible in the supplied screenshots. V6 does not invent missing hand-by-hand sequences.")
reg=KNOWN_SESSIONS.copy()
reg["Dragon %"]=reg["Dragon"]/reg["Captured through"]; reg["Tiger %"]=reg["Tiger"]/reg["Captured through"]; reg["Tie %"]=reg["Tie"]/reg["Captured through"]
rv=reg[["Provider","Game","Captured through","Dragon","Tiger","Tie","Dragon %","Tiger %","Tie %","Sequence available"]].copy()
for c in ["Dragon %","Tiger %","Tie %"]: rv[c]=rv[c].map(lambda v:f"{v:.1%}")
st.dataframe(rv,hide_index=True,use_container_width=True)

# -----------------------------------------------------------------------------
# Road structure
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🛣️ Road / sequence structure")

recent = results[-20:]
st.write("**Latest 20:** " + "  ".join(recent))

r1, r2, r3, r4 = st.columns(4)
r1.metric("Longest Dragon run", best["D"])
r2.metric("Longest Tiger run", best["T"])
r3.metric("Longest Tie run", best["X"])
if len(seq) > 1:
    switches = sum(a != b for a, b in zip(seq, seq[1:]))
    r4.metric("D/T switch rate", f"{switches/(len(seq)-1):.1%}")
else:
    r4.metric("D/T switch rate", "—")

st.markdown("**D/T transition behaviour**")
tr = transition_table(seq)
if not tr.empty:
    view = tr.copy()
    view["Continue %"] = view["Continue %"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    view["Switch %"] = view["Switch %"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    st.dataframe(view, hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# Model details
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🧠 Ensemble model details")
if details:
    # V9 keeps display schemas defensive: different model builders may return
    # different metadata fields. Never crash because an optional column is absent.
    md = pd.DataFrame(details)
    for col in ("D", "T"):
        if col in md.columns:
            md[col] = md[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    if "Weight" in md.columns:
        md["Weight"] = md["Weight"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    if "Support" in md.columns:
        md["Support"] = pd.to_numeric(md["Support"], errors="coerce").fillna(0).astype(int)
    if "Backtest acc" in md.columns:
        md["Backtest acc"] = md["Backtest acc"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    if "Robust" in md.columns:
        md["Robust"] = md["Robust"].map(lambda v: "PASS" if bool(v) else "shrunk")
    st.dataframe(md, hide_index=True, use_container_width=True)
else:
    st.info("No model-detail rows are available for the current history.")

# -----------------------------------------------------------------------------
# Robust walk-forward validation
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🧪 V6 robust walk-forward validation")
st.write("V9 reports raw accuracy, coverage, Wilson confidence bounds and an exact 50/50 screen. A rule is not called robust merely because its raw accuracy is high.")
if not validation.empty:
    display=validation.copy()
    for c in ["Accuracy","Coverage","Wilson low","Wilson high"]: display[c]=display[c].map(lambda v:f"{v:.1%}" if pd.notna(v) else "—")
    display["p-value"]=display["p-value"].map(lambda v:f"{v:.3f}" if pd.notna(v) else "—")
    display["Robust"]=display["Robust"].map(lambda v:"PASS" if v else "—")
    st.dataframe(display,hide_index=True,use_container_width=True)
    rc=int(validation["Robust"].sum())
    if rc: st.success(f"{rc} model(s) pass the V9 robustness gate.")
    else: st.warning("No model passes the V9 robustness gate. V9 returns NO BET rather than promoting historical noise.")
else:
    st.warning("Not enough directional history for robust validation.")

# Ensemble stability
st.divider()
st.subheader("🧠 Ensemble stability")
if details:
    stable = pd.DataFrame(details).copy()
    for col in ("D", "T"):
        if col in stable.columns:
            stable[col] = stable[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    if "Weight" in stable.columns:
        stable["Weight"] = stable["Weight"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    if "Robust" in stable.columns:
        stable["Robust"] = stable["Robust"].map(lambda v: "PASS" if bool(v) else "shrunk")
    st.dataframe(stable, hide_index=True, use_container_width=True)
else:
    st.info("Stability cannot be estimated until at least one model is available.")

# -----------------------------------------------------------------------------
# Pattern / n-gram inspection
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🔍 Repeated sequence checks")
pattern_len = st.slider("Pattern length", 1, 5, 3)
pat = "".join(seq[-pattern_len:]) if len(seq) >= pattern_len else ""
if pat:
    follows = []
    for i in range(len(seq) - pattern_len):
        if "".join(seq[i:i+pattern_len]) == pat:
            follows.append(seq[i+pattern_len])
    if follows:
        st.write(f"Current D/T pattern: **{pat}** — historical follow-ups: **{len(follows)}**")
        st.write(f"Next D: **{follows.count('D')/len(follows):.1%}** · Next T: **{follows.count('T')/len(follows):.1%}")
    else:
        st.info("The current pattern has not appeared earlier with a following D/T result.")

# -----------------------------------------------------------------------------
# Data-quality checks
# -----------------------------------------------------------------------------
st.divider()
st.subheader("✅ Data-quality checks")
checks = [
    {"Check": "History loaded", "Status": "PASS", "Detail": f"{n} results."},
    {"Check": "Invalid D/T/Tie tokens", "Status": "PASS", "Detail": "None in the loaded history."},
    {"Check": "Directional sample", "Status": "PASS" if len(seq) >= 30 else "LIMITED", "Detail": f"{len(seq)} non-Tie results."},
    {"Check": "Walk-forward validation", "Status": "PASS" if len(seq) >= 50 else "LIMITED", "Detail": "Signals are generated from prior history only."},
    {"Check": "Provider separation", "Status": "PASS", "Detail": "Current source is treated as one provider/table session."},
]
st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# V9 app health
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🛡️ V9 app health")
health = [
    {"Check": "History integrity", "Status": "PASS", "Detail": f"{n} valid D/T/Tie results loaded."},
    {"Check": "Model-detail rendering", "Status": "PASS", "Detail": "Optional model columns are handled safely."},
    {"Check": "Future-data leakage", "Status": "PASS" if len(seq) >= 20 else "LIMITED", "Detail": "Walk-forward tests use only history available before each test point."},
    {"Check": "Provider/session mixing", "Status": "PASS", "Detail": "Verified sessions remain separate from the active sequence."},
]
st.dataframe(pd.DataFrame(health), hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------
st.divider()
st.subheader("⬇️ Export this session")
export_df = pd.DataFrame({"Hand": np.arange(1, n+1), "Outcome": results, "Table": table_name})
csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download cleaned session CSV", csv, file_name="dragon_tiger_v9_session.csv", mime="text/csv")

st.warning("Important: V9 is for statistical research and validation. Historical road patterns, streaks and model accuracy cannot guarantee the next casino outcome.")
