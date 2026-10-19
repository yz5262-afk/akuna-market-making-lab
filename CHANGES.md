# Public-copy fixes

- Replace the nonexistent `numpy.minimize` with `scipy.optimize.minimize` so
  calibration can run instead of silently retaining fallback parameters.
- Retain fallback behavior if the rate optimizer reports non-convergence.
- Use a local seeded RandomState for pricing, preserving the prior draw convention
  without resetting the caller's global NumPy generator.
- Use the actual clipped change at the zero rate floor when computing equity
  returns, matching the source market-transition model at that boundary.
- Add a synthetic diagnostic example and focused validation cases. These changes
  do not establish strategy profitability or full competition-engine parity.
