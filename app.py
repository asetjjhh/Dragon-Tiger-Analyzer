import re
from collections import Counter
from math import sqrt

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dragon Tiger Analyzer V2",
    page_icon="🐉",
    layout="wide",
)

st.title("🐉🐯 Dragon Tiger Analyzer")
st.caption(
    "V2 • Card-level history, rank/suit tracking and shoe-aware statistics. "
    "Analytical tool — not a guaranteed prediction system."
)

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
RANK_VALUE = {r: i + 1 for i, r in enumerate(RANKS)}
SUITS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}

def normalize_rank(token):
    t = token.strip().upper()
    aliases = {"ACE": "A", "JACK": "J", "QUEEN": "Q", "KING": "K"}
    return aliases.get(t, t)

def parse_card(token):
    token = token.strip().upper()
    if not token:
        return None

    # Accept AS, 10H, QD, or rank only (A, 10, Q).
    m = re.fullmatch(r"(10|[2-9AJQK])([SHDC])?", token)
    if not m:
        return None

    rank = normalize_rank(m.group(1))
    suit = m.group(2)
    return {"rank": rank, "suit": suit, "raw": token}

def parse_card_lines(text):
    # One card per line is the safest mobile format.
    tokens = []
    for line in text.replace(",", "\n").splitlines():
        value = line.strip()
        if value:
            tokens.append(value)

    cards = []
    invalid = []
    for token in tokens:
        card = parse_card(token)
        if card:
            cards.append(card)
        else:
            invalid.append(token)
    return cards, invalid

def parse_results(text):
    tokens = [
        x.strip().upper()
        for x in text.replace(",", " ").replace("\n", " ").split()
    ]
    mapping = {"DRAGON": "D", "TIGER": "T", "TIE": "X"}
    out = []
    for token in tokens:
        token = mapping.get(token, token)
        if token in {"D", "T", "X"}:
            out.append(token)
    return out

def outcome_from_cards(dragon, tiger):
    dv = RANK_VALUE[dragon["rank"]]
    tv = RANK_VALUE[tiger["rank"]]
    if dv > tv:
        return "D"
    if tv > dv:
        return "T"
    return "X"

def streak_info(results):
    if not results:
        return "-", 0, 0, 0

    cur = results[-1]
    current = 0
    for x in reversed(results):
        if x == cur:
            current += 1
        else:
            break

    dmax = tmax = 0
    d = t = 0
    for x in results:
        if x == "D":
            d += 1
            t = 0
        elif x == "T":
            t += 1
            d = 0
        else:
            d = t = 0
        dmax = max(dmax, d)
        tmax = max(tmax, t)

    return cur, current, dmax, tmax

def transition_stats(results):
    seq = [x for x in results if x in ("D", "T")]
    if len(seq) < 2:
        return 0.0, 0, 0
    transitions = len(seq) - 1
    switches = sum(a != b for a, b in zip(seq, seq[1:]))
    return switches / transitions, switches, transitions

def conditional_probability(results, pattern):
    seq = [x for x in results if x in ("D", "T")]
    pattern = list(pattern)
    hits = next_d = next_t = 0
    L = len(pattern)

    if not pattern or len(seq) <= L:
        return 0, 0, 0

    for i in range(len(seq) - L):
        if seq[i:i + L] == pattern:
            hits += 1
            if seq[i + L] == "D":
                next_d += 1
            elif seq[i + L] == "T":
                next_t += 1

    return hits, next_d, next_t

def two_sided_z(obs, expected, n):
    if n == 0 or expected <= 0 or expected >= 1:
        return np.nan
    return (obs - expected) / sqrt(expected * (1 - expected) / n)

def exact_shoe_probabilities(cards, decks):
    """Conditional next-hand probabilities from known remaining cards.

    Valid only when the shoe composition is known and cards include suits.
    """
    total = 52 * decks
    if len(cards) >= total:
        return {"D": 0.0, "T": 0.0, "X": 0.0, "remaining": 0}

    known = Counter((c["rank"], c["suit"]) for c in cards)
    rank_remaining = {rank: 4 * decks for rank in RANKS}

    for (rank, suit), count in known.items():
        if suit in SUITS:
            rank_remaining[rank] -= count

    remaining = sum(rank_remaining.values())
    if remaining < 2:
        return {"D": 0.0, "T": 0.0, "X": 0.0, "remaining": remaining}

    d = t = x = 0.0
    for dr in RANKS:
        for tr in RANKS:
            ways = rank_remaining[dr] * rank_remaining[tr]
            if dr == tr:
                ways -= rank_remaining[dr]
            ways = max(0, ways)
            if RANK_VALUE[dr] > RANK_VALUE[tr]:
                d += ways
            elif RANK_VALUE[tr] > RANK_VALUE[dr]:
                t += ways
            else:
                x += ways

    denom = remaining * (remaining - 1)
    return {
        "D": d / denom,
        "T": t / denom,
        "X": x / denom,
        "remaining": remaining,
    }

def card_key(card):
    return (card["rank"], card["suit"])

def card_label(card):
    return f'{card["rank"]}{card["suit"] or ""}'

# ---------- sidebar ----------
st.sidebar.header("V2 data input")
mode = st.sidebar.radio(
    "Input mode",
    ["Card-level", "Quick result"],
    index=0,
)

decks = st.sidebar.selectbox(
    "Deck model for shoe tracking",
    [1, 2, 4, 6, 8],
    index=4,
    help="Use this only when the table really uses a fixed shoe with the selected number of decks.",
)

recent_window = st.sidebar.selectbox("Recent window", [10, 20, 30, 50], index=1)
pattern = (
    st.sidebar.text_input("Conditional pattern", "TTT")
    .strip()
    .upper()
    .replace(" ", "")
)

card_history = []
results = []
invalid = []
card_mode_ready = False

if mode == "Card-level":
    st.sidebar.markdown(
        "**Mobile format:** enter one card per line, oldest → newest."
    )
    dragon_text = st.sidebar.text_area(
        "Dragon cards",
        value="AS\n7H\n10D\nKC\n5S\n9H",
        height=180,
        help="Examples: AS, 10H, QD, K. Suits are S/H/D/C. Rank-only cards are allowed, but exact shoe tracking needs suits.",
    )
    tiger_text = st.sidebar.text_area(
        "Tiger cards",
        value="7C\n9D\n10C\n4H\nQS\n2D",
        height=180,
        help="Enter the same number of Tiger cards as Dragon cards.",
    )

    dragon_cards, dragon_invalid = parse_card_lines(dragon_text)
    tiger_cards, tiger_invalid = parse_card_lines(tiger_text)
    invalid = dragon_invalid + tiger_invalid

    if dragon_invalid:
        st.sidebar.error(f"Invalid Dragon card(s): {', '.join(dragon_invalid)}")
    if tiger_invalid:
        st.sidebar.error(f"Invalid Tiger card(s): {', '.join(tiger_invalid)}")

    if len(dragon_cards) != len(tiger_cards):
        st.warning(
            f"Dragon has {len(dragon_cards)} parsed card(s), while Tiger has "
            f"{len(tiger_cards)}. Enter one Dragon and one Tiger card for every hand."
        )
    elif dragon_cards and not invalid:
        card_mode_ready = True
        for i, (dragon, tiger) in enumerate(zip(dragon_cards, tiger_cards), start=1):
            card_history.append(
                {
                    "Hand": i,
                    "Dragon": card_label(dragon),
                    "Tiger": card_label(tiger),
                    "Dragon value": RANK_VALUE[dragon["rank"]],
                    "Tiger value": RANK_VALUE[tiger["rank"]],
                    "Outcome": outcome_from_cards(dragon, tiger),
                    "Suited Tie": (
                        outcome_from_cards(dragon, tiger) == "X"
                        and dragon["suit"] is not None
                        and tiger["suit"] is not None
                        and dragon["suit"] == tiger["suit"]
                    ),
                }
            )
        results = [row["Outcome"] for row in card_history]

else:
    raw = st.sidebar.text_area(
        "Enter results oldest → newest",
        value="D T D D T T D T D D T T D T D",
        height=180,
        help="Use D = Dragon, T = Tiger, X = Tie. Dragon/Tiger/Tie also work.",
    )
    results = parse_results(raw)

if not results:
    st.info("Enter at least one hand to start the analysis.")
    st.stop()

# ---------- summary ----------
counts = pd.Series(results).value_counts()
d = int(counts.get("D", 0))
t = int(counts.get("T", 0))
x = int(counts.get("X", 0))
n = len(results)

recent = results[-min(recent_window, n):]
rd = recent.count("D")
rt = recent.count("T")
rx = recent.count("X")

cur, cur_n, dmax, tmax = streak_info(results)
switch_rate, switches, transitions = transition_stats(results)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hands", n)
c2.metric("Dragon", f"{d/n:.1%}")
c3.metric("Tiger", f"{t/n:.1%}")
c4.metric("Tie", f"{x/n:.1%}")
c5.metric("Current", f"{cur} × {cur_n}")

# ---------- card-level panel ----------
if mode == "Card-level" and card_mode_ready:
    st.divider()
    st.subheader("🃏 Card-level history")

    card_df = pd.DataFrame(card_history)
    st.dataframe(
        card_df[["Hand", "Dragon", "Tiger", "Dragon value", "Tiger value", "Outcome", "Suited Tie"]],
        hide_index=True,
        use_container_width=True,
    )

    suited_ties = int(card_df["Suited Tie"].sum())
    exact_suited_rate = suited_ties / n if n else 0

    a, b, c, dcol = st.columns(4)
    a.metric("Known cards", 2 * n)
    b.metric("Suited Ties", suited_ties)
    c.metric("Suited Tie rate", f"{exact_suited_rate:.2%}")
    dcol.metric("Unique exact cards", card_df.shape[0] * 2)

    # Duplicate check: an exact card can appear at most `decks` times in a fixed multi-deck shoe.
    all_cards = []
    for row in card_history:
        all_cards.append(row["Dragon"])
        all_cards.append(row["Tiger"])

    parsed_all = []
    for token in all_cards:
        card = parse_card(token)
        if card and card["suit"]:
            parsed_all.append(card)

    if len(parsed_all) == 2 * n:
        exact_counts = Counter(card_key(c) for c in parsed_all)
        impossible = {
            f"{r}{s}": count
            for (r, s), count in exact_counts.items()
            if count > decks
        }
        if impossible:
            st.error(
                "Exact-card count exceeds the selected deck model: "
                + ", ".join(f"{k} × {v}" for k, v in impossible.items())
            )
        else:
            st.success(
                f"Exact suited-card tracking is internally consistent with a {decks}-deck model."
            )
    else:
        st.info(
            "Some cards do not include suits. Rank analysis still works, but exact shoe depletion "
            "cannot be fully verified."
        )

    # Rank distribution
    rank_rows = []
    for rank in RANKS:
        rank_rows.append(
            {
                "Rank": rank,
                "Dragon": sum(c["rank"] == rank for c in dragon_cards),
                "Tiger": sum(c["rank"] == rank for c in tiger_cards),
                "Total": sum(c["rank"] == rank for c in dragon_cards + tiger_cards),
            }
        )
    st.markdown("**Observed rank distribution**")
    st.dataframe(pd.DataFrame(rank_rows), hide_index=True, use_container_width=True)

    # Shoe-aware probabilities require every entered card to have a suit.
    all_input_cards = dragon_cards + tiger_cards
    full_suits = all(c["suit"] is not None for c in all_input_cards)
    if full_suits:
        shoe = exact_shoe_probabilities(all_input_cards, decks)
        st.markdown("### 🧮 Known-shoe next-hand baseline")
        st.caption(
            "This is a mathematical conditional baseline only if the selected fixed shoe, "
            "deck count and all removed cards are known. It is not a prediction guarantee."
        )
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Remaining cards", shoe["remaining"])
        p2.metric("Dragon", f'{shoe["D"]:.2%}')
        p3.metric("Tiger", f'{shoe["T"]:.2%}')
        p4.metric("Tie", f'{shoe["X"]:.2%}')
    else:
        st.info(
            "Add suits to every card (for example AS, 10H, QC) to unlock exact known-shoe "
            "depletion probabilities."
        )

# ---------- descriptive stats ----------
st.divider()
left, right = st.columns(2)

with left:
    st.subheader("📊 Historical statistics")
    stats = pd.DataFrame(
        {
            "Outcome": ["Dragon", "Tiger", "Tie"],
            "Count": [d, t, x],
            "Observed %": [d / n, t / n, x / n],
        }
    )
    stats["Observed %"] = stats["Observed %"].map(lambda v: f"{v:.2%}")
    st.dataframe(stats, hide_index=True, use_container_width=True)

    st.write(f"Longest Dragon streak: **{dmax}**")
    st.write(f"Longest Tiger streak: **{tmax}**")
    st.write(
        f"Alternation/switch rate (D/T only): **{switch_rate:.2%}** "
        f"({switches}/{transitions} transitions)"
    )

with right:
    st.subheader(f"🔎 Recent {len(recent)} hands")
    rstats = pd.DataFrame(
        {
            "Outcome": ["Dragon", "Tiger", "Tie"],
            "Count": [rd, rt, rx],
            "Observed %": [
                rd / len(recent),
                rt / len(recent),
                rx / len(recent),
            ],
        }
    )
    rstats["Observed %"] = rstats["Observed %"].map(lambda v: f"{v:.2%}")
    st.dataframe(rstats, hide_index=True, use_container_width=True)
    st.write("Recent sequence:")
    st.code(" ".join(recent))

# ---------- current card details ----------
if mode == "Card-level" and card_mode_ready:
    st.divider()
    latest = card_history[-1]
    st.subheader("🎴 Latest hand")
    q1, q2, q3 = st.columns(3)
    q1.metric("Dragon card", latest["Dragon"])
    q2.metric("Tiger card", latest["Tiger"])
    q3.metric("Result", latest["Outcome"])
    if latest["Suited Tie"]:
        st.success("♠♥♦♣ This hand is a suited Tie (same rank + same suit).")

# ---------- conditional probability ----------
st.divider()
st.subheader("🧮 Conditional pattern analysis")

if pattern and all(ch in "DT" for ch in pattern):
    hits, next_d, next_t = conditional_probability(results, pattern)
    if hits:
        st.write(f"After **{pattern}**, the next result was observed **{hits}** time(s).")
        a, b, c = st.columns(3)
        a.metric("Next Dragon", f"{next_d / hits:.1%}")
        b.metric("Next Tiger", f"{next_t / hits:.1%}")
        c.metric("Occurrences", hits)
        st.info(
            "Historical conditional frequency is not the same as a true probability "
            "for the next casino hand."
        )
    else:
        st.warning(
            "That pattern has not occurred enough in the entered data to estimate "
            "a conditional frequency."
        )
else:
    st.warning("Pattern must contain only D and T, e.g. D, TT, DTD, or TTT.")

# ---------- baseline comparison ----------
st.divider()
st.subheader("⚖️ Baseline comparison")

st.write(
    "Dragon and Tiger are symmetric in the standard game. This screen uses a "
    "50/50 non-Tie comparison for descriptive testing. Exact table rules, payouts "
    "and variants should be configured separately."
)

non_tie = d + t
if non_tie:
    d_non_tie = d / non_tie
    z = two_sided_z(d_non_tie, 0.5, non_tie)
    st.metric("Observed Dragon share among non-Ties", f"{d_non_tie:.2%}")
    if np.isfinite(z):
        st.write(f"Approximate z-score vs 50/50: **{z:.2f}**")
    st.caption(
        "A z-score alone does not establish predictive ability. Multiple testing "
        "and small samples can produce false discoveries."
    )

# ---------- signal ----------
st.divider()
st.subheader("🎯 Transparent analytical signal")

recent_non_tie = [q for q in recent if q in ("D", "T")]
if len(recent_non_tie) >= 10:
    rd_nt = recent_non_tie.count("D") / len(recent_non_tie)
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
    "This signal is intentionally descriptive. It is NOT a claim that the next "
    "hand has that percentage chance of winning and does not overcome casino house edge."
)

# ---------- export ----------
st.divider()
st.subheader("💾 Export data")

if mode == "Card-level" and card_mode_ready:
    export_df = pd.DataFrame(card_history)
    export_name = "dragon_tiger_card_history_v2.csv"
else:
    export_df = pd.DataFrame({"hand": range(1, n + 1), "result": results})
    export_name = "dragon_tiger_results_v2.csv"

st.download_button(
    "Download CSV",
    export_df.to_csv(index=False).encode("utf-8"),
    export_name,
    "text/csv",
)

with st.expander("V2 developer notes"):
    st.write(
        """
V2 additions:
- Actual Dragon/Tiger card input
- Rank and suit parsing
- Result derived automatically from card ranks
- Suited-Tie detection
- Exact-card duplicate validation for a selected fixed deck model
- Rank distribution table
- Known-shoe conditional baseline when every suit is known
- Card-level CSV export

Next:
V3 = road/chart engine
V4 = backtesting
V5 = statistical validation
V6 = model comparison and calibration
V7 = persistent database/dashboard
V8 = screenshot/OCR-assisted entry
"""
    )
