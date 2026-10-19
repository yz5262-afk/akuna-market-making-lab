# Validation

Checked on October 19, 2026 with Python 3.13.

- Six unittest methods passed: expiry payoff boundaries, shared-path strike monotonicity, NumPy RNG isolation, the zero-rate floor, quote invariants/inventory skew, and FOK inventory/cash/price gates.
- Ran the complete synthetic diagnostic example with 401 generated history observations. Calibration replaced the fallback prior, the strike curve was monotone, and quote diagnostics/figure were generated.
- All Python files passed syntax checks. No credentials, private paths or notebook outputs are included.

## Limits

No official competition harness, leaderboard comparison, backtest P&L, live exchange, latency measurement or external statistical calibration benchmark was run. The synthetic example is a reproducibility and behavior check.
