import re
from collections import Counter
from math import sqrt
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dragon Tiger Analyzer V4", page_icon="🐉", layout="wide")

st.title("🐉🐯 Dragon Tiger Analyzer — V4")
st.caption("V4 • Separate table/session histories, road analysis, leakage-safe walk-forward backtesting, signal validation and error checks.")

# IMPORTANT: Do not hard-code a provider's card ranking unless its rules are verified.
# V4 treats D/T/X history as authoritative for sequence analysis. Card-derived ranking is optional.
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}


def parse_result(x):
    x = str(x).strip().upper()
    return {
        "D": "D", "DRAGON": "D",
        "T": "T", "TIGER": "T",
        "X": "X", "TIE": "X",
    }.get(x)


def parse_card(token):
    token = str(token).strip().upper()
    m = re.fullmatch(r"(10|[2-9AJQK])([SHDC])?", token)
    if not m:
        return None
    return {"rank": m.group(1), "suit": m.group(2)}


def clean_df(df):
    df = df.copy()
    cols = {str(c).lower().strip(): c for c in df.columns}
    if "outcome" not in cols:
        return pd.DataFrame(), "CSV must contain an Outcome column."

    out = pd.DataFrame()
    if "table" in cols:
        out["Table"] = df[cols["table"]].fillna("Unknown Table").astype(str).str.strip()
    elif "session" in cols:
        out["Table"] = df[cols["session"]].fillna("Unknown Session").astype(str).str.strip()
    else:
        out["Table"] = "Table 1"

    if "hand" in cols:
        out["Hand"] = pd.to_numeric(df[cols["hand"]], errors="coerce")
    else:
        out["Hand"] = np.arange(1, len(df) + 1)

    out["Outcome"] = df[cols["outcome"]].map(parse_result)
    out["Dragon"] = df[cols["dragon"]].fillna("").astype(str).str.upper().str.strip() if "dragon" in cols else ""
    out["Tiger"] = df[cols["tiger"]].fillna("").astype(str).str.upper().str.strip() if "tiger" in cols else ""

    bad = int(out["Outcome"].isna().sum())
    out = out[out["Outcome"].notna()].reset_index(drop=True)
    return out, (f"Ignored {bad} invalid outcome row(s)." if bad else "")


def sequence_only(results):
    return [r for r in results if r in ("D", "T", "X")]


def dt_only(results):
    return [r for r in results if r in ("D", "T")]


def streak_info(results):
    best = {"D": 0, "T": 0, "X": 0}
    cur = {"D": 0, "T": 0, "X": 0}
    for r in results:
        for k in cur:
            cur[k] = cur[k] + 1 if r == k else 0
            best[k] = max(best[k], cur[k])
    current = results[-1] if results else "-"
    current_n = 0
    for r in reversed(results):
        if r != current:
            break
        current_n += 1
    return current, current_n, best


def transition_counts(results):
    seq = dt_only(results)
    mat = pd.DataFrame(0, index=["D", "T"], columns=["D", "T"])
    for a, b in zip(seq, seq[1:]):
        mat.loc[a, b] += 1
    return mat


def transition_rates(results):
    seq = dt_only(results)
    rows = []
    for source in ("D", "T"):
        nxt = [b for a, b in zip(seq, seq[1:]) if a == source]
        rows.append({
            "After": source,
            "Next D": nxt.count("D"),
            "Next T": nxt.count("T"),
            "Continue rate": (nxt.count(source) / len(nxt)) if nxt else np.nan,
            "Switch rate": (nxt.count("T" if source == "D" else "D") / len(nxt)) if nxt else np.nan,
            "Samples": len(nxt),
        })
    return pd.DataFrame(rows)


def run_rule(hist, rule, window):
    if len(hist) < window:
        return "LEAVE"
    h = hist[-window:]
    share_d = h.count("D") / window
    if rule == "recent_majority":
        if share_d > 0.5:
            return "D"
        if share_d < 0.5:
            return "T"
        return "LEAVE"
    if rule == "threshold":
        if share_d >= 0.60:
            return "D"
        if share_d <= 0.40:
            return "T"
        return "LEAVE"
    if rule == "trend_follow":
        if len(h) < 4:
            return "LEAVE"
        a, b = h[-2], h[-1]
        if a == b == "D":
            return "D"
        if a == b == "T":
            return "T"
        return "LEAVE"
    return "LEAVE"


def walk_forward(results, window=10, rule="threshold", include_ties_as_miss=False):
    seq = dt_only(results)
    rows = []
    if len(seq) <= window:
        return pd.DataFrame()
    for i in range(window, len(seq)):
        hist = seq[:i]
        signal = run_rule(hist, rule, window)
        actual = seq[i]
        if signal in ("D", "T"):
            correct = signal == actual
        else:
            correct = np.nan
        rows.append({
            "Test hand": i + 1,
            "History used": i,
            "Signal": signal,
            "Actual": actual,
            "Correct": correct,
        })
    return pd.DataFrame(rows)


def score_backtest(bt):
    if bt.empty:
        return None
    sig = bt[bt["Signal"].isin(["D", "T"])].copy()
    if sig.empty:
        return {"signals": 0, "accuracy": np.nan, "coverage": 0.0, "d": 0, "t": 0}
    return {
        "signals": len(sig),
        "accuracy": float(sig["Correct"].mean()),
        "coverage": len(sig) / len(bt),
        "d": int((sig["Signal"] == "D").sum()),
        "t": int((sig["Signal"] == "T").sum()),
    }


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - adj) / den, (centre + adj) / den


def conditional_pattern(results, pattern):
    seq = dt_only(results)
    L = len(pattern)
    hits = d = t = 0
    for i in range(len(seq) - L):
        if "".join(seq[i:i + L]) == pattern:
            hits += 1
            if seq[i + L] == "D":
                d += 1
            else:
                t += 1
    return hits, d, t


def road_grid(results, width=18):
    vals = sequence_only(results)
    if not vals:
        return pd.DataFrame()
    rows = (len(vals) + width - 1) // width
    grid = [["" for _ in range(width)] for _ in range(rows)]
    for i, v in enumerate(vals):
        grid[i // width][i % width] = v
    return pd.DataFrame(grid, columns=[str(i + 1) for i in range(width)])


def parse_manual(text):
    tokens = [x for x in re.split(r"[\s,;|]+", text.strip()) if x]
    parsed = [parse_result(x) for x in tokens]
    invalid = [tokens[i] for i, v in enumerate(parsed) if v is None]
    vals = [v for v in parsed if v is not None]
    return vals, invalid


def make_manual_df(vals, table_name):
    return pd.DataFrame({
        "Table": table_name,
        "Hand": np.arange(1, len(vals) + 1),
        "Outcome": vals,
        "Dragon": "",
        "Tiger": "",
    })

# ---------------- Sidebar ----------------
st.sidebar.header("V4 controls")
st.sidebar.info("Sequence analysis uses D/T/X. Card/rank analysis is optional and is never allowed to silently override the recorded outcome.")

upload = st.sidebar.file_uploader("Optional CSV history", type=["csv"])
manual = st.sidebar.text_area("Paste D/T/X results, oldest → newest", "", height=140, placeholder="D T T D X D T ...")
manual_table = st.sidebar.text_input("Manual table/session name", "Current Session")

window = st.sidebar.selectbox("Backtest window", [5, 10, 15, 20, 30, 50], index=1)
rule = st.sidebar.selectbox(
    "Backtest rule",
    ["threshold", "recent_majority", "trend_follow"],
    format_func=lambda x: {
        "threshold": "Recent 60/40 threshold",
        "recent_majority": "Recent majority",
        "trend_follow": "Two-result trend follow",
    }[x],
)
pattern = st.sidebar.text_input("Pattern to test (D/T only)", "TTT").replace(" ", "").upper()

if upload is not None:
    try:
        raw = pd.read_csv(upload)
        data, note = clean_df(raw)
        if note:
            st.sidebar.warning(note)
        if data.empty:
            st.error("No valid history was found in the CSV.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    vals, invalid = parse_manual(manual)
    if invalid:
        st.sidebar.error("Unrecognized result(s): " + ", ".join(invalid[:12]))
    if not vals:
        st.info("Paste a history or upload a CSV to begin V4.")
        st.stop()
    data = make_manual_df(vals, manual_table.strip() or "Current Session")

# ---------------- Table/session selection ----------------
tables = list(data["Table"].astype(str).unique())
selected = st.sidebar.selectbox("Table / session", tables)
tab = data[data["Table"].astype(str) == str(selected)].copy().reset_index(drop=True)
results = tab["Outcome"].tolist()

if not results:
    st.warning("No valid results in the selected table/session.")
    st.stop()

# ---------------- Header summary ----------------
counts = Counter(results)
d, t, x = counts.get("D", 0), counts.get("T", 0), counts.get("X", 0)
n = len(results)
cur, cur_n, best = streak_info(results)
recent_n = min(window, n)
recent = results[-recent_n:]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("History", n)
c2.metric("Dragon", f"{d / n:.1%}")
c3.metric("Tiger", f"{t / n:.1%}")
c4.metric("Tie", f"{x / n:.1%}")
c5.metric("Current", f"{cur} × {cur_n}")
st.caption(f"Selected table/session: **{selected}**")

# ---------------- Signal ----------------
st.divider()
st.subheader("🎯 V4 signal")

seq_dt = dt_only(results)
if len(seq_dt) < window:
    signal = "LEAVE"
    reason = f"Need at least {window} non-Tie results for the selected rule; only {len(seq_dt)} are available."
else:
    signal = run_rule(seq_dt, rule, window)
    if signal == "D":
        reason = f"The selected rule currently leans Dragon using the latest {window} non-Tie results."
    elif signal == "T":
        reason = f"The selected rule currently leans Tiger using the latest {window} non-Tie results."
    else:
        reason = "The selected rule does not produce a directional signal."

s1, s2 = st.columns([1, 2])
with s1:
    st.metric("Current signal", {"D": "DRAGON", "T": "TIGER", "LEAVE": "NO BET"}[signal])
with s2:
    st.write(reason)
    st.caption("A signal is a historical rule output, not a guarantee of the next hand.")

# ---------------- Road / sequence ----------------
st.divider()
st.subheader("🛣️ Sequence and road behaviour")
road = road_grid(results)
if not road.empty:
    st.dataframe(road, hide_index=True, use_container_width=True)

switches = sum(a != b for a, b in zip(seq_dt, seq_dt[1:]))
transitions = max(0, len(seq_dt) - 1)
if transitions:
    st.write(f"D/T switch rate: **{switches / transitions:.1%}** ({switches}/{transitions})")
else:
    st.write("Not enough non-Tie results for switch analysis.")

r1, r2, r3 = st.columns(3)
r1.write(f"Longest Dragon streak: **{best['D']}**")
r2.write(f"Longest Tiger streak: **{best['T']}**")
r3.write(f"Longest Tie streak: **{best['X']}**")
st.write("Recent sequence: **" + " ".join(recent) + "**")

st.markdown("**Transition counts**")
st.dataframe(transition_counts(results), use_container_width=True)
st.markdown("**Transition rates**")
tr = transition_rates(results).copy()
tr["Continue rate"] = tr["Continue rate"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
tr["Switch rate"] = tr["Switch rate"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
st.dataframe(tr, hide_index=True, use_container_width=True)

# ---------------- Recent structure ----------------
st.divider()
a, b = st.columns(2)
with a:
    st.subheader(f"📊 Latest {recent_n}")
    rc = Counter(recent)
    rdf = pd.DataFrame({
        "Outcome": ["D", "T", "X"],
        "Count": [rc["D"], rc["T"], rc["X"]],
        "Share": [rc["D"] / recent_n, rc["T"] / recent_n, rc["X"] / recent_n],
    })
    rdf["Share"] = rdf["Share"].map(lambda v: f"{v:.2%}")
    st.dataframe(rdf, hide_index=True, use_container_width=True)
with b:
    st.subheader("🔍 Pattern test")
    if pattern and all(c in "DT" for c in pattern):
        hits, nd, nt = conditional_pattern(results, pattern)
        if hits:
            st.write(f"**{pattern}** occurred {hits} time(s) with a following non-Tie result.")
            p1, p2 = st.columns(2)
            p1.metric("Next Dragon", f"{nd / hits:.1%}")
            p2.metric("Next Tiger", f"{nt / hits:.1%}")
            lo, hi = wilson_interval(max(nd, nt), hits)
            st.caption(f"Small-sample interval for the larger observed share: {lo:.1%}–{hi:.1%}. This is descriptive, not proof of predictability.")
        else:
            st.warning("No qualifying occurrence of this pattern.")
    else:
        st.warning("Use only D and T, for example TT, DTD or TTT.")

# ---------------- Backtesting ----------------
st.divider()
st.subheader("🧪 Walk-forward backtesting")
st.write("V4 generates each historical signal using only results that occurred before that test hand. The future result is revealed only for scoring.")

bt = walk_forward(results, window=window, rule=rule)
if bt.empty:
    st.warning(f"Not enough non-Tie history to backtest a {window}-result window.")
else:
    score = score_backtest(bt)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test points", len(bt))
    m2.metric("Signals", score["signals"])
    m3.metric("Signal accuracy", f"{score['accuracy']:.1%}" if score["signals"] else "—")
    m4.metric("Coverage", f"{score['coverage']:.1%}")
    if score["signals"]:
        lo, hi = wilson_interval(round(score["accuracy"] * score["signals"]), score["signals"])
        st.caption(f"Approximate 95% Wilson interval for historical signal accuracy: {lo:.1%}–{hi:.1%}. Small samples can be highly unstable.")
    st.dataframe(bt.tail(100), hide_index=True, use_container_width=True)

# ---------------- Walk-forward comparison ----------------
st.subheader("📚 Rule comparison")
comparison = []
for w in [5, 10, 15, 20, 30, 50]:
    if len(seq_dt) <= w:
        continue
    for rname in ["threshold", "recent_majority", "trend_follow"]:
        test = walk_forward(results, window=w, rule=rname)
        sc = score_backtest(test)
        comparison.append({
            "Window": w,
            "Rule": rname,
            "Signals": sc["signals"],
            "Accuracy": sc["accuracy"],
            "Coverage": sc["coverage"],
        })
if comparison:
    comp = pd.DataFrame(comparison)
    comp["Accuracy"] = comp["Accuracy"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    comp["Coverage"] = comp["Coverage"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    st.dataframe(comp, hide_index=True, use_container_width=True)
else:
    st.info("Need more non-Tie history before rule comparison is useful.")

# ---------------- Card validation ----------------nst.divider()
st.subheader("🃏 Optional card validation")
st.caption("Card analysis is deliberately conservative. Because provider rules can differ, V4 does not assume Ace-high or Ace-low without a verified table rule.")

card_rows = []
for _, row in tab.iterrows():
    dc = parse_card(row.get("Dragon", ""))
    tc = parse_card(row.get("Tiger", ""))
    if dc and tc:
        card_rows.append({
            "Hand": row["Hand"],
            "Dragon": row["Dragon"],
            "Tiger": row["Tiger"],
            "Recorded": row["Outcome"],
            "Dragon suit": SUITS.get(dc["suit"], "") if dc["suit"] else "",
            "Tiger suit": SUITS.get(tc["suit"], "") if tc["suit"] else "",
            "Same suit": bool(dc["suit"] and tc["suit"] and dc["suit"] == tc["suit"]),
        })
if card_rows:
    st.dataframe(pd.DataFrame(card_rows), hide_index=True, use_container_width=True)
    st.info("No rank-derived winner is computed here until the exact provider/table card-ranking rule is verified.")
else:
    st.info("If you later add Dragon/Tiger card columns to the CSV, V4 will validate the cards and identify suited ties without overriding the recorded outcome.")

# ---------------- Data quality ----------------
st.divider()
st.subheader("✅ Data-quality checks")
quality = []
quality.append({"Check": "Valid outcomes", "Status": "PASS", "Detail": f"{n} recorded D/T/Tie results in this selection."})
quality.append({"Check": "Non-Tie sample", "Status": "PASS" if len(seq_dt) >= 20 else "LIMITED", "Detail": f"{len(seq_dt)} non-Tie results available."})
quality.append({"Check": "Backtest sample", "Status": "PASS" if len(bt) >= 50 else "LIMITED", "Detail": f"{len(bt)} walk-forward test points."})
quality.append({"Check": "Future leakage", "Status": "PROTECTED", "Detail": "Backtest signals use only prior results."})
st.dataframe(pd.DataFrame(quality), hide_index=True, use_container_width=True)

# ---------------- Export ----------------
st.divider()
st.subheader("⬇️ Export cleaned history")
csv = tab.to_csv(index=False).encode("utf-8")
st.download_button("Download selected table/session CSV", csv, file_name=f"dragon_tiger_{re.sub(r'[^A-Za-z0-9_-]+', '_', str(selected))}.csv", mime="text/csv")

st.warning("V4 is a research/backtesting tool. Historical sequences and road patterns do not guarantee the next casino outcome. Use NO BET/LEAVE when evidence is weak.")
