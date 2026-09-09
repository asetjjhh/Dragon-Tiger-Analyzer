import re
from collections import Counter
from math import sqrt
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dragon Tiger Analyzer V3", page_icon="🐉", layout="wide")

st.title("🐉🐯 Dragon Tiger Analyzer — V3")
st.caption("V3 • Separate tables/sessions, road engine, card/rank analysis, conditional testing and walk-forward validation.")

# Locked project rule: Ace is always the lowest rank in this project.
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
RANK_VALUE = {r: i + 1 for i, r in enumerate(RANKS)}
SUITS = {"S":"♠", "H":"♥", "D":"♦", "C":"♣"}


def parse_card(token):
    token = str(token).strip().upper()
    m = re.fullmatch(r"(10|[2-9AJQK])([SHDC])?", token)
    if not m:
        return None
    return {"rank": m.group(1), "suit": m.group(2)}


def outcome_from_cards(dragon, tiger):
    dv, tv = RANK_VALUE[dragon["rank"]], RANK_VALUE[tiger["rank"]]
    return "D" if dv > tv else "T" if tv > dv else "X"


def parse_result(x):
    x = str(x).strip().upper()
    return {"DRAGON":"D", "TIGER":"T", "TIE":"X"}.get(x, x if x in {"D","T","X"} else None)


def clean_df(df):
    df = df.copy()
    cols = {c.lower().strip(): c for c in df.columns}
    if "outcome" not in cols:
        return pd.DataFrame()
    out = pd.DataFrame()
    if "table" in cols:
        out["Table"] = df[cols["table"]].astype(str)
    elif "session" in cols:
        out["Table"] = df[cols["session"]].astype(str)
    else:
        out["Table"] = "Table 1"
    out["Hand"] = pd.to_numeric(df[cols["hand"]], errors="coerce") if "hand" in cols else np.arange(1, len(df)+1)
    out["Outcome"] = df[cols["outcome"]].map(parse_result)
    out["Dragon"] = df[cols["dragon"]].astype(str).str.upper() if "dragon" in cols else ""
    out["Tiger"] = df[cols["tiger"]].astype(str).str.upper() if "tiger" in cols else ""
    out = out[out["Outcome"].notna()].reset_index(drop=True)
    return out


def road_grid(results, width=12):
    vals = [x for x in results if x in ("D","T","X")]
    if not vals:
        return pd.DataFrame()
    rows = (len(vals) + width - 1) // width
    grid = [["" for _ in range(width)] for _ in range(rows)]
    for i, v in enumerate(vals):
        grid[i // width][i % width] = v
    return pd.DataFrame(grid, columns=[str(i+1) for i in range(width)])


def streaks(results):
    best = {"D":0,"T":0,"X":0}
    cur = {"D":0,"T":0,"X":0}
    for r in results:
        for k in cur: cur[k] = cur[k] + 1 if r == k else 0
        for k in best: best[k] = max(best[k], cur[k])
    current = results[-1] if results else "-"
    current_n = 0
    for r in reversed(results):
        if r != current: break
        current_n += 1
    return current, current_n, best


def transition_matrix(results):
    seq = [x for x in results if x in ("D","T")]
    mat = pd.DataFrame(0, index=["D","T"], columns=["D","T"])
    for a,b in zip(seq, seq[1:]): mat.loc[a,b] += 1
    return mat


def conditional(results, pattern):
    seq = [x for x in results if x in ("D","T")]
    L = len(pattern)
    hits = d = t = 0
    for i in range(len(seq)-L):
        if "".join(seq[i:i+L]) == pattern:
            hits += 1
            if seq[i+L] == "D": d += 1
            else: t += 1
    return hits, d, t


def wilson_lower(p, n, z=1.96):
    if n == 0: return 0.0
    den = 1 + z*z/n
    centre = p + z*z/(2*n)
    adj = z*sqrt((p*(1-p)+z*z/(4*n))/n)
    return (centre-adj)/den


def walk_forward(results, window=20, threshold=0.60, min_signal_n=10):
    """Backtest a deliberately simple, pre-declared recent-majority rule.
    It predicts only D/T; ties are scored separately as non-directional.
    No future information is used.
    """
    seq = [x for x in results if x in ("D","T")]
    rows=[]
    for i in range(min_signal_n, len(seq)):
        hist = seq[max(0,i-window):i]
        if len(hist) < min_signal_n: continue
        share = hist.count("D")/len(hist)
        if share >= threshold: signal="D"
        elif share <= 1-threshold: signal="T"
        else: signal="LEAVE"
        actual=seq[i]
        correct = signal in ("D","T") and signal == actual
        rows.append({"Test index":i+1,"History n":len(hist),"Dragon share":share,"Signal":signal,"Actual":actual,"Correct":correct if signal!="LEAVE" else np.nan})
    return pd.DataFrame(rows)


def z_score(p, expected, n):
    if n <= 0 or expected <= 0 or expected >= 1: return np.nan
    return (p-expected)/sqrt(expected*(1-expected)/n)

# ---------------- Sidebar ----------------
st.sidebar.header("V3 controls")
st.sidebar.info("Project rule: Ace = 1 (lowest). This setting is intentionally locked.")

upload = st.sidebar.file_uploader("Optional CSV history", type=["csv"])
manual = st.sidebar.text_area("Quick results (D/T/X), oldest → newest", "", height=120)
window = st.sidebar.selectbox("Recent window", [10,20,30,50,100], index=1)
pattern = st.sidebar.text_input("Conditional pattern", "TTT").replace(" ", "").upper()

if upload is not None:
    try:
        raw_df = pd.read_csv(upload)
        data = clean_df(raw_df)
        if data.empty: st.error("CSV must contain an Outcome column. Optional columns: Table, Hand, Dragon, Tiger.")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        data = pd.DataFrame()
else:
    vals = [parse_result(x) for x in re.split(r"[\s,;]+", manual.strip()) if x.strip()]
    vals = [x for x in vals if x]
    data = pd.DataFrame({"Table":"Current Session", "Hand":range(1,len(vals)+1), "Outcome":vals, "Dragon":"", "Tiger":""})

if data.empty:
    st.info("Start with a CSV or paste D/T/X results. V3 can analyze multiple tables independently when the CSV has a Table column.")
    st.stop()

# ---------------- Table selector ----------------
tables = list(data["Table"].dropna().astype(str).unique())
selected = st.sidebar.selectbox("Table / session", tables)
tab = data[data["Table"].astype(str) == str(selected)].copy().reset_index(drop=True)
results = tab["Outcome"].tolist()

# ---------------- Summary ----------------
counts = Counter(results)
d,t,x = counts.get("D",0), counts.get("T",0), counts.get("X",0)
n = len(results)
cur, cur_n, best = streaks(results)
recent = results[-min(window,n):]
non_tie = [r for r in recent if r in ("D","T")]

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Hands", n)
c2.metric("Dragon", f"{d/n:.1%}")
c3.metric("Tiger", f"{t/n:.1%}")
c4.metric("Tie", f"{x/n:.1%}")
c5.metric("Current streak", f"{cur} × {cur_n}")

st.caption(f"Selected table/session: **{selected}**")

# ---------------- Road engine ----------------nst.divider()
st.subheader("🛣️ Road / sequence engine")
st.dataframe(road_grid(results), hide_index=True, use_container_width=True)

seq = [r for r in results if r in ("D","T")]
switches = sum(a != b for a,b in zip(seq,seq[1:]))
transitions = max(0,len(seq)-1)
st.write(f"D/T-only switch rate: **{switches/transitions:.1%}** ({switches}/{transitions})" if transitions else "Not enough D/T results for switch-rate analysis.")

m1,m2 = st.columns(2)
with m1:
    st.write(f"Longest Dragon streak: **{best['D']}**")
    st.write(f"Longest Tiger streak: **{best['T']}**")
with m2:
    st.write(f"Longest Tie streak: **{best['X']}**")
    st.write("Recent sequence: **" + " ".join(recent) + "**")

st.markdown("**D/T transition counts**")
st.dataframe(transition_matrix(results), use_container_width=True)

# ---------------- Recent + baseline ----------------
st.divider()
a,b = st.columns(2)
with a:
    st.subheader(f"📊 Recent {len(recent)}")
    rc = Counter(recent)
    st.dataframe(pd.DataFrame({"Outcome":["D","T","X"],"Count":[rc['D'],rc['T'],rc['X']],"Share":[rc['D']/len(recent),rc['T']/len(recent),rc['X']/len(recent)]}).assign(Share=lambda q:q.Share.map(lambda z:f"{z:.2%}")), hide_index=True, use_container_width=True)
with b:
    st.subheader("⚖️ 50/50 non-Tie check")
    nt = d+t
    if nt:
        share=d/nt
        z=z_score(share,0.5,nt)
        st.metric("Dragon share among non-Ties",f"{share:.2%}")
        st.write(f"Approximate z-score: **{z:.2f}**")
        st.caption("This tests historical imbalance; it does not prove the next hand is predictable.")
    else: st.info("No non-Tie hands yet.")

# ---------------- Conditional ----------------
st.divider()
st.subheader("🧮 Conditional pattern testing")
if pattern and all(c in "DT" for c in pattern):
    hits,nd,nt = conditional(results,pattern)
    if hits:
        st.write(f"Pattern **{pattern}** occurred **{hits}** time(s) with a following D/T result.")
        q1,q2,q3=st.columns(3)
        q1.metric("Next Dragon",f"{nd/hits:.1%}")
        q2.metric("Next Tiger",f"{nt/hits:.1%}")
        q3.metric("Occurrences",hits)
        if hits < 20: st.warning("Small sample: treat this as descriptive only.")
    else: st.warning("Pattern has no qualifying historical occurrences in this table/session.")
else: st.warning("Use only D and T in the pattern, e.g. TT, DTD, TTT.")

# ---------------- Card/rank engine ----------------nst.divider()
st.subheader("🃏 Card / rank engine")
card_rows=[]
for _,row in tab.iterrows():
    dc=parse_card(row.get("Dragon",""))
    tc=parse_card(row.get("Tiger",""))
    if dc and tc:
        derived=outcome_from_cards(dc,tc)
        card_rows.append({"Hand":row["Hand"],"Dragon":row["Dragon"],"Tiger":row["Tiger"],"D value":RANK_VALUE[dc['rank']],"T value":RANK_VALUE[tc['rank']],"Derived":derived,"Recorded":row["Outcome"],"Match":derived==row["Outcome"],"Suited Tie":derived=="X" and dc['suit'] and dc['suit']==tc['suit']})
if card_rows:
    cdf=pd.DataFrame(card_rows)
    st.dataframe(cdf, hide_index=True, use_container_width=True)
    mismatches=int((~cdf["Match"]).sum())
    if mismatches:
        st.error(f"{mismatches} card/result mismatch(es). Check the recorded result and table rules before using rank-derived analysis.")
    else:
        st.success("All entered card-derived outcomes match the recorded outcomes under Ace = 1.")
    st.write("**Rank distribution**")
    ranks=[]
    for r in RANKS:
        ranks.append({"Rank":r,"Dragon":sum(parse_card(x).get('rank')==r for x in tab['Dragon'] if parse_card(x)),"Tiger":sum(parse_card(x).get('rank')==r for x in tab['Tiger'] if parse_card(x))})
    st.dataframe(pd.DataFrame(ranks),hide_index=True,use_container_width=True)
else:
    st.info("Add Dragon/Tiger cards in the CSV to activate card-level validation and rank analysis.")

# ---------------- Walk-forward ----------------nst.divider()
st.subheader("🧪 Walk-forward validation")
st.caption("V3 tests a pre-declared recent-majority rule only on outcomes that occur after the historical window. It never uses future hands to create the signal.")
wf_window = st.selectbox("Backtest history window", [10,20,30,50], index=1, key="wfwin")
wf = walk_forward(results, window=wf_window, threshold=0.60, min_signal_n=10)
if wf.empty:
    st.warning("Not enough historical D/T data for walk-forward testing. Keep collecting complete histories naturally; do not force extra data just for the app.")
else:
    actionable=wf[wf.Signal.isin(["D","T"])].copy()
    leave_rate=(wf.Signal=="LEAVE").mean()
    b1,b2,b3,b4=st.columns(4)
    b1.metric("Test points",len(wf))
    b2.metric("LEAVE rate",f"{leave_rate:.1%}")
    b3.metric("Actionable points",len(actionable))
    if len(actionable):
        acc=actionable.Correct.mean()
        b4.metric("Actionable hit rate",f"{acc:.1%}")
        st.write(f"Wilson 95% lower bound: **{wilson_lower(acc,len(actionable)):.1%}**")
    st.dataframe(wf.tail(100), hide_index=True, use_container_width=True)
    st.warning("A backtest result is not evidence of a guaranteed edge. It can be unstable, overfit, or produced by chance.")

# ---------------- Signal ----------------nst.divider()
st.subheader("🎯 Conservative analytical state")
if len(non_tie) < 10:
    signal="⚪ LEAVE"
    reason=f"Only {len(non_tie)} recent non-Tie results; minimum sample not met."
else:
    share=non_tie.count("D")/len(non_tie)
    if share>=0.60:
        signal="🐉 DRAGON LEAN"; reason=f"Recent non-Tie sample is Dragon-heavy ({share:.1%})."
    elif share<=0.40:
        signal="🐯 TIGER LEAN"; reason=f"Recent non-Tie sample is Tiger-heavy ({1-share:.1%})."
    else:
        signal="⚪ LEAVE"; reason=f"Recent non-Tie sample is near balanced ({share:.1%} Dragon)."
st.markdown(f"## {signal}")
st.write(reason)
st.caption("This is a descriptive research state, not a claim that the next casino hand can be predicted reliably.")

# ---------------- Export ----------------nst.divider()
st.subheader("📥 Export")
st.download_button("Download selected table CSV", tab.to_csv(index=False).encode("utf-8"), file_name=f"dragon_tiger_{str(selected).replace(' ','_')}.csv", mime="text/csv")

st.success("V3 loaded successfully. Keep different tables/sessions separate; use larger histories for validation rather than assuming short patterns predict the next hand.")
