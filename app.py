import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt

st.set_page_config(page_title="Dragon Tiger Analyzer", page_icon="🐉", layout="wide")

st.title("🐉🐯 Dragon Tiger Analyzer")
st.caption("Statistical analysis and backtesting tool — not a guaranteed prediction system.")

# ---------- helpers ----------
def parse_results(text):
    tokens = [x.strip().upper() for x in text.replace(",", " ").replace("\n", " ").split()]
    mapping = {"DRAGON":"D", "TIGER":"T", "TIE":"X"}
    out = []
    for t in tokens:
        t = mapping.get(t, t)
        if t in {"D", "T", "X"}:
            out.append(t)
    return out

def streak_info(results):
    if not results:
        return "-", 0, 0, 0
    cur = results[-1]
    n = 0
    for x in reversed(results):
        if x == cur:
            n += 1
        else:
            break
    dmax = tmax = 0
    d = t = 0
    for x in results:
        if x == "D":
            d += 1; t = 0
        elif x == "T":
            t += 1; d = 0
        else:
            d = t = 0
        dmax = max(dmax, d); tmax = max(tmax, t)
    return cur, n, dmax, tmax

def transition_stats(results):
    seq = [x for x in results if x in ("D","T")]
    if len(seq) < 2:
        return 0, 0, 0
    transitions = len(seq)-1
    switches = sum(a != b for a,b in zip(seq, seq[1:]))
    return switches/transitions, switches, transitions

def conditional_probability(results, pattern):
    seq = [x for x in results if x in ("D","T")]
    pattern = list(pattern)
    hits = next_d = next_t = 0
    L = len(pattern)
    for i in range(len(seq)-L):
        if seq[i:i+L] == pattern:
            hits += 1
            nxt = seq[i+L]
            if nxt == "D": next_d += 1
            elif nxt == "T": next_t += 1
    return hits, next_d, next_t

def baseline_se(p, n):
    return sqrt(p*(1-p)/n) if n else 0

def two_sided_z(obs, expected, n):
    if n == 0 or expected <= 0 or expected >= 1:
        return np.nan
    return (obs-expected)/sqrt(expected*(1-expected)/n)

# ---------- sidebar ----------
st.sidebar.header("Data input")
raw = st.sidebar.text_area(
    "Enter results oldest → newest",
    value="D T D D T T D T D D T T D T D",
    height=180,
    help="Use D = Dragon, T = Tiger, X = Tie. You can also type Dragon/Tiger/Tie."
)
results = parse_results(raw)

window = st.sidebar.selectbox("Recent window", [10, 20, 30, 50], index=1)
pattern = st.sidebar.text_input("Conditional pattern", "TTT").strip().upper().replace(" ", "")

if not results:
    st.warning("Enter at least one D, T, or X result.")
    st.stop()

# ---------- summary ----------
counts = pd.Series(results).value_counts()
d = int(counts.get("D",0)); t = int(counts.get("T",0)); x = int(counts.get("X",0))
n = len(results)
non_tie = d+t
recent = results[-min(window,n):]
rd = recent.count("D"); rt = recent.count("T"); rx = recent.count("X")

cur, cur_n, dmax, tmax = streak_info(results)
switch_rate, switches, transitions = transition_stats(results)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Hands", n)
c2.metric("Dragon", f"{d/n:.1%}" if n else "—")
c3.metric("Tiger", f"{t/n:.1%}" if n else "—")
c4.metric("Tie", f"{x/n:.1%}" if n else "—")
c5.metric("Current", f"{cur} × {cur_n}")

st.divider()

# ---------- descriptive stats ----------
left, right = st.columns(2)

with left:
    st.subheader("📊 Historical statistics")
    stats = pd.DataFrame({
        "Outcome": ["Dragon","Tiger","Tie"],
        "Count": [d,t,x],
        "Observed %": [d/n, t/n, x/n]
    })
    stats["Observed %"] = stats["Observed %"].map(lambda v: f"{v:.2%}")
    st.dataframe(stats, hide_index=True, use_container_width=True)

    st.write(f"Longest Dragon streak: **{dmax}**")
    st.write(f"Longest Tiger streak: **{tmax}**")
    st.write(f"Alternation/switch rate (D/T only): **{switch_rate:.2%}**")

with right:
    st.subheader(f"🔎 Recent {len(recent)} hands")
    rstats = pd.DataFrame({
        "Outcome": ["Dragon","Tiger","Tie"],
        "Count": [rd,rt,rx],
        "Observed %": [rd/len(recent), rt/len(recent), rx/len(recent)]
    })
    rstats["Observed %"] = rstats["Observed %"].map(lambda v: f"{v:.2%}")
    st.dataframe(rstats, hide_index=True, use_container_width=True)

    st.write("Recent sequence:")
    st.code(" ".join(recent))

# ---------- conditional probability ----------
st.divider()
st.subheader("🧮 Conditional pattern analysis")

if pattern and all(ch in "DT" for ch in pattern):
    hits, next_d, next_t = conditional_probability(results, pattern)
    if hits:
        st.write(f"After **{pattern}**, the next result was observed {hits} time(s).")
        a,b,c = st.columns(3)
        a.metric("Next Dragon", f"{next_d/hits:.1%}")
        b.metric("Next Tiger", f"{next_t/hits:.1%}")
        c.metric("Occurrences", hits)
        st.info("This is an observed historical frequency, not a guaranteed next-hand probability.")
    else:
        st.warning("That pattern has not occurred often enough in the entered data to estimate a conditional frequency.")
else:
    st.warning("Pattern must contain only D and T, e.g. D, TT, DTD, or TTT.")

# ---------- baseline comparison ----------
st.divider()
st.subheader("⚖️ Baseline comparison")

st.write(
    "For the common 8-deck mathematical model, Dragon and Tiger are symmetric. "
    "This screen uses an illustrative non-tie baseline of 50% Dragon / 50% Tiger; "
    "the exact rules and payouts of your table should be configured separately."
)

if non_tie:
    d_non_tie = d/non_tie
    z = two_sided_z(d_non_tie, 0.5, non_tie)
    st.metric("Observed Dragon share among non-Ties", f"{d_non_tie:.2%}")
    if np.isfinite(z):
        st.write(f"Approximate z-score vs 50/50: **{z:.2f}**")
    st.caption("A z-score alone does not prove predictive ability; repeated testing can create false discoveries.")

# ---------- simple transparent signal ----------
st.divider()
st.subheader("🎯 Transparent analytical signal")

recent_non_tie = [q for q in recent if q in ("D","T")]
if len(recent_non_tie) >= 10:
    rd_nt = recent_non_tie.count("D") / len(recent_non_tie)
    # Deliberately conservative: this is a descriptive signal, not a claimed winning probability.
    if rd_nt >= 0.60:
        signal = "🐉 DRAGON LEAN"
        reason = f"Recent non-Tie sample is Dragon-heavy ({rd_nt:.1%})."
    elif rd_nt <= 0.40:
        signal = "🐯 TIGER LEAN"
        reason = f"Recent non-Tie sample is Tiger-heavy ({1-rd_nt:.1%})."
    else:
        signal = "⚪ LEAVE"
        reason = f"Recent non-Tie sample is near balanced ({rd_nt:.1%} Dragon)."
else:
    signal = "⚪ LEAVE"
    reason = "Fewer than 10 recent non-Tie outcomes."

st.markdown(f"## {signal}")
st.write(reason)
st.warning(
    "This MVP signal is intentionally descriptive. It is NOT a claim that the next hand has "
    "that percentage chance of winning, and it does not overcome the casino house edge."
)

# ---------- export ----------
st.divider()
st.subheader("💾 Export data")
df = pd.DataFrame({"hand": range(1,n+1), "result": results})
st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode("utf-8"),
    "dragon_tiger_results.csv",
    "text/csv"
)

with st.expander("Developer notes"):
    st.write("""
    Next versions can add:
    - card-level input (Dragon rank/suit, Tiger rank/suit)
    - shoe/deck tracking when valid information is available
    - provider/variant separation
    - road visualization
    - pattern backtesting
    - train/validation/out-of-sample testing
    - calibrated model probabilities
    - model performance, ROI and drawdown tracking
    - screenshot/OCR-assisted data entry
    """)
