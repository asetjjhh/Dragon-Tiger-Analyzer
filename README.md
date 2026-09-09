# 🐉🐯 Dragon Tiger Analyzer — V4

V4 is a research and validation tool for Dragon Tiger history.

## V4 changes
- Keeps tables/sessions separate.
- Accepts D/T/X history directly, so suits/cards are optional.
- Road/sequence analysis: streaks, switching, transitions, recent structure.
- Conditional pattern testing.
- Leakage-safe walk-forward backtesting.
- Compares several simple pre-declared rules and windows.
- Shows coverage as well as accuracy so a model cannot hide behind very few signals.
- Data-quality checks and CSV export.
- Optional card/suit validation without silently assuming provider-specific Ace ranking.
- Conservative **DRAGON / TIGER / NO BET** signal.

## Recommended workflow
1. Keep each provider/game variant/table session separate.
2. If a table ends around 75–160 hands, save/export that session and start a new session.
3. You do not need to manually build a 500-hand single-table history.
4. Use screenshots for human review and enter only the D/T/Tie sequence when convenient.
5. Use V4 backtesting to test whether a rule survives historical walk-forward testing.

## Data format
CSV should contain at least:

- `Outcome` — D, T or X
- optional `Table` or `Session`
- optional `Hand`
- optional `Dragon` card
- optional `Tiger` card

## Important
V4 does not assume Ace-high or Ace-low for card-derived winner calculations. Provider/table rules must be verified first. The recorded D/T/Tie result remains authoritative for sequence analysis.

V4 is not a guaranteed prediction system and must not be treated as proof that the next casino outcome is predictable.
