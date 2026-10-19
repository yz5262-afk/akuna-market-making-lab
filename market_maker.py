import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final

AJARAI_NAME: Final[str] = "AJR"
AJARAI_UNDERLYING_ID: Final[int] = 2
FED_FUNDS_RATE_NAME: Final[str] = "FED"
FED_FUNDS_RATE_UNDERLYING_ID: Final[int] = 1
RATE_STRIKE_GRID: Final[float] = 0.25
THERIODIC_NAME: Final[str] = "THR"
THERIODIC_UNDERLYING_ID: Final[int] = 3

UNDERLYING_NAME_BY_ID: Final[dict[int, str]] = {
    AJARAI_UNDERLYING_ID: AJARAI_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID: FED_FUNDS_RATE_NAME,
    THERIODIC_UNDERLYING_ID: THERIODIC_NAME,
}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class BinaryOption:
    legs: "tuple[OptionLeg, ...]"
    option_id: int
    steps_until_expiry: int
    strike: float

    def __post_init__(self) -> None:
        if self.steps_until_expiry < 0:
            raise ValueError("Steps until expiry must be non-negative")

        if not self.legs:
            raise ValueError("Binary option must have at least one leg")

        underlying_ids: list[int] = [leg.underlying_id for leg in self.legs]
        if len(underlying_ids) != len(set(underlying_ids)):
            raise ValueError("Binary option legs must reference distinct underlyings")

        if any(leg.weight == 0 for leg in self.legs):
            raise ValueError("Binary option leg weights must be non-zero")

    def __str__(self) -> str:
        terms: list[str] = []
        for index, leg in enumerate(self.legs):
            name: str = UNDERLYING_NAME_BY_ID.get(leg.underlying_id, str(leg.underlying_id))
            magnitude: float = abs(leg.weight)
            magnitude_str: str = "" if magnitude == 1 else f"{magnitude:.2f}*"
            if index == 0:
                sign: str = "-" if leg.weight < 0 else ""
            else:
                sign = " - " if leg.weight < 0 else " + "
            terms.append(f"{sign}{magnitude_str}{name}")
        observable_expression: str = "".join(terms)
        return f"{self.option_id} ({self.steps_until_expiry}d {observable_expression} >= {self.strike:.2f})"

    def advance_step(self) -> "BinaryOption":
        if self.steps_until_expiry == 0:
            return self

        return replace(self, steps_until_expiry=self.steps_until_expiry - 1)

    def contract_matches(self, other: "BinaryOption") -> bool:
        return replace(other, option_id=self.option_id) == self

    def expiry_valuation(self, value_by_underlying_id: dict[int, float]) -> float:
        return 1.0 if self.observable_value(value_by_underlying_id) >= self.strike else 0.0

    def observable_value(self, value_by_underlying_id: dict[int, float]) -> float:
        return sum(leg.weight * value_by_underlying_id[leg.underlying_id] for leg in self.legs)


@dataclass(frozen=True)
class FokOrder:
    counterparty_id: int
    option_id: int
    order_type: "OrderType"
    price: float
    quantity: int

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("FOK order price must be non-negative")

        if self.quantity <= 0:
            raise ValueError("FOK order quantity must be positive")


@dataclass(frozen=True)
class MarketHistory:
    values_by_underlying_id: dict[int, tuple[float, ...]]

    def __post_init__(self) -> None:
        lengths: set[int] = {len(values) for values in self.values_by_underlying_id.values()}
        if len(lengths) > 1:
            raise ValueError("All underlyings must have the same number of historical days")

        if lengths and next(iter(lengths)) <= 0:
            raise ValueError("Market history must contain at least one day")

    @property
    def num_days(self) -> int:
        if not self.values_by_underlying_id:
            return 0
        return len(next(iter(self.values_by_underlying_id.values())))


@dataclass(frozen=True)
class MarketParameters:
    ajarai_drift: float
    ajarai_idio_std_dev: float
    ajarai_rate_beta: float
    ajarai_sector_beta: float
    rate_down_probability: float
    rate_reversion_strength: float
    rate_up_probability: float
    sector_std_dev: float
    theriodic_drift: float
    theriodic_idio_std_dev: float
    theriodic_rate_beta: float
    theriodic_sector_beta: float

    rate_step: float = 0.25
    rate_target: float = 2.0

    def __post_init__(self) -> None:
        if self.rate_step <= 0:
            raise ValueError("Rate step must be positive")

        if self.rate_up_probability <= 0 or self.rate_down_probability <= 0:
            raise ValueError("Rate up/down probabilities must both be positive")

        if self.rate_up_probability + self.rate_down_probability > 1:
            raise ValueError("Rate up/down probabilities must not sum to more than 1")

        if self.rate_target < 0:
            raise ValueError("Rate target must be non-negative")

        if not (0 <= self.rate_reversion_strength <= 1):
            raise ValueError("Rate reversion strength must be between 0 and 1")

        if self.ajarai_idio_std_dev < 0 or self.theriodic_idio_std_dev < 0 or self.sector_std_dev < 0:
            raise ValueError("Standard deviations must be non-negative")

    def advance_company_value(
        self,
        current_value: float,
        rate_change: float,
        sector_shock: float,
        *,
        drift: float,
        rate_beta: float,
        sector_beta: float,
        idio_std_dev: float,
    ) -> float:
        idiosyncratic_shock: float = random.gauss(mu=0.0, sigma=idio_std_dev)
        log_return: float = drift + (rate_beta * rate_change) + (sector_beta * sector_shock) + idiosyncratic_shock
        return round(current_value * math.exp(log_return), 2)

    def advance_rate(self, rate_value: float) -> float:
        up_probability, down_probability = self.tilted_rate_probabilities(rate_value)
        draw: float = random.random()
        if draw < up_probability:
            return self.next_rate_value(rate_value, 1)

        if draw < up_probability + down_probability:
            return self.next_rate_value(rate_value, -1)

        return rate_value

    def advance_step(self, value_by_underlying_id: dict[int, float]) -> dict[int, float]:
        current_rate_value: float = value_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID]
        rate_value: float = self.advance_rate(current_rate_value)
        rate_change: float = round(rate_value - current_rate_value, 2)
        sector_shock: float = random.gauss(mu=0.0, sigma=self.sector_std_dev)
        return {
            FED_FUNDS_RATE_UNDERLYING_ID: rate_value,
            AJARAI_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[AJARAI_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.ajarai_drift,
                rate_beta=self.ajarai_rate_beta,
                sector_beta=self.ajarai_sector_beta,
                idio_std_dev=self.ajarai_idio_std_dev,
            ),
            THERIODIC_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[THERIODIC_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.theriodic_drift,
                rate_beta=self.theriodic_rate_beta,
                sector_beta=self.theriodic_sector_beta,
                idio_std_dev=self.theriodic_idio_std_dev,
            ),
        }

    def next_rate_value(self, rate_value: float, num_grid_steps: int) -> float:
        return max(round(rate_value + num_grid_steps * self.rate_step, 2), 0.0)

    def tilted_rate_probabilities(self, rate_value: float) -> tuple[float, float]:
        tilt: float = self.rate_reversion_strength * (self.rate_target - rate_value)
        up_probability: float = min(max(self.rate_up_probability + tilt, 0.0), 1.0)
        down_probability: float = min(max(self.rate_down_probability - tilt, 0.0), 1.0 - up_probability)
        return up_probability, down_probability


@dataclass(frozen=True)
class OptionLeg:
    underlying_id: int
    weight: float


class OrderType(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Position:
    def __init__(self) -> None:
        self.option_quantity_by_option_id: dict[int, int] = defaultdict(int)

    def add_option_quantity(self, option_id: int, quantity: int) -> None:
        self.option_quantity_by_option_id[option_id] += quantity


@dataclass(frozen=True)
class Quote:
    bid_price: float
    bid_quantity: int
    offer_price: float
    offer_quantity: int

    def __post_init__(self) -> None:
        if self.bid_quantity <= 0 or self.offer_quantity <= 0:
            raise ValueError("Quote quantities must be positive")

        if not (0.0 <= self.bid_price <= 1.0 and 0.0 <= self.offer_price <= 1.0):
            raise ValueError("Quote prices must be between 0 and 1")

        if self.bid_price >= self.offer_price:
            raise ValueError("Quote bid price must be less than offer price")

        if any(abs(round(price * 100) - price * 100) > 1e-6 for price in (self.bid_price, self.offer_price)):
            raise ValueError("Quote prices must be in whole pennies (multiples of 0.01)")


@dataclass(frozen=True)
class Underlying:
    name: str
    underlying_id: int
    value: float

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Underlying):
            return False
        return self.underlying_id == other.underlying_id

# ============================================================================
# YOUR MARKET MAKER -- fill in the six stubbed methods below
# ============================================================================
import numpy as np
from scipy.optimize import minimize

class MarketMaker:

    def __init__(
        self,
        underlying_initial_state: list[Underlying],
        option_initial_state: list[BinaryOption],
        cash_balance: float,
    ) -> None:

        self.underlying_state: list[Underlying] = underlying_initial_state
        self.active_option_state: list[BinaryOption] = option_initial_state
        self.cash_balance: float = cash_balance
        self.position: Position = Position()

        # Fallback prior parameters used if warm_up fails.
        self.estimated_parameters: MarketParameters = MarketParameters(
            ajarai_drift=0.0005,
            ajarai_idio_std_dev=0.015,
            ajarai_rate_beta=-0.01,
            ajarai_sector_beta=1.0,
            rate_down_probability=0.2,
            rate_reversion_strength=0.1,
            rate_up_probability=0.2,
            sector_std_dev=0.01,
            theriodic_drift=0.0005,
            theriodic_idio_std_dev=0.015,
            theriodic_rate_beta=-0.01,
            theriodic_sector_beta=1.0,
        )

        # Hard inventory and cash-risk limits.
        self.MAX_INVENTORY = 20
        self.MIN_CASH_BUFFER = 10.0

    @property
    def name(self) -> str:
        return "AlphaQuant_Robust_MM"

    def warm_up(self, market_history: MarketHistory) -> None:
        """
        Estimate MarketParameters from historical observations.
        All estimation is protected by a fallback in case of failure.
        """
        try:
            fed_rates = np.array(
                market_history.values_by_underlying_id[
                    FED_FUNDS_RATE_UNDERLYING_ID
                ]
            )

            ajr_vals = np.array(
                market_history.values_by_underlying_id[
                    AJARAI_UNDERLYING_ID
                ]
            )

            thr_vals = np.array(
                market_history.values_by_underlying_id[
                    THERIODIC_UNDERLYING_ID
                ]
            )

            # 1. FED rate maximum likelihood estimation.

            r_curr = fed_rates[:-1]
            diffs = np.round(
                fed_rates[1:] - r_curr,
                2,
            )

            moves = np.zeros_like(
                diffs,
                dtype=int,
            )

            moves[diffs > 0.01] = 1
            moves[diffs < -0.01] = -1

            def fed_neg_ll(params):
                p_up_base, p_down_base, strength = params

                tilt = strength * (2.0 - r_curr)

                p_up = np.clip(
                    p_up_base + tilt,
                    0.0,
                    1.0,
                )

                p_down = np.clip(
                    p_down_base - tilt,
                    0.0,
                    1.0 - p_up,
                )

                p_stay = np.maximum(
                    0.0,
                    1.0 - p_up - p_down,
                )

                probs = np.where(
                    moves == 1,
                    p_up,
                    np.where(
                        moves == -1,
                        p_down,
                        p_stay,
                    ),
                )

                return -np.sum(
                    np.log(
                        np.maximum(
                            probs,
                            1e-9,
                        )
                    )
                )

            res_fed = minimize(
                fed_neg_ll,
                x0=[0.2, 0.2, 0.1],
                bounds=[
                    (0.001, 0.499),
                    (0.001, 0.499),
                    (0.0, 1.0),
                ],
                method="L-BFGS-B",
            )

            if not res_fed.success:
                raise ValueError("Rate-parameter optimization did not converge")
            p_up_est, p_down_est, strength_est = res_fed.x

            # 2. AI company log-return OLS and residual PCA.

            delta_r = np.round(
                fed_rates[1:] - fed_rates[:-1],
                2,
            )

            r_ajr = np.log(
                ajr_vals[1:] / ajr_vals[:-1]
            )

            r_thr = np.log(
                thr_vals[1:] / thr_vals[:-1]
            )

            X = np.column_stack(
                [
                    np.ones_like(delta_r),
                    delta_r,
                ]
            )

            coeff_ajr, _, _, _ = np.linalg.lstsq(
                X,
                r_ajr,
                rcond=None,
            )

            coeff_thr, _, _, _ = np.linalg.lstsq(
                X,
                r_thr,
                rcond=None,
            )

            e_ajr = (
                r_ajr
                - (
                    coeff_ajr[0]
                    + coeff_ajr[1] * delta_r
                )
            )

            e_thr = (
                r_thr
                - (
                    coeff_thr[0]
                    + coeff_thr[1] * delta_r
                )
            )

            E = np.column_stack(
                [
                    e_ajr,
                    e_thr,
                ]
            )

            cov_E = np.cov(
                E,
                rowvar=False,
            )

            # PCA extraction with covariance-matrix decomposition.

            eigenvalues, eigenvectors = np.linalg.eigh(
                cov_E
            )

            sector_shocks = (
                E
                @ eigenvectors[
                    :,
                    np.argmax(eigenvalues),
                ]
            )

            sector_std_dev = np.std(
                sector_shocks,
                ddof=1,
            )

            X_s = sector_shocks[
                :,
                np.newaxis,
            ]

            beta_ajr_s, _, _, _ = np.linalg.lstsq(
                X_s,
                e_ajr,
                rcond=None,
            )

            beta_thr_s, _, _, _ = np.linalg.lstsq(
                X_s,
                e_thr,
                rcond=None,
            )

            self.estimated_parameters = MarketParameters(
                ajarai_drift=float(
                    coeff_ajr[0]
                ),
                ajarai_idio_std_dev=float(
                    np.std(
                        e_ajr
                        - beta_ajr_s[0]
                        * sector_shocks,
                        ddof=1,
                    )
                ),
                ajarai_rate_beta=float(
                    coeff_ajr[1]
                ),
                ajarai_sector_beta=float(
                    beta_ajr_s[0]
                ),
                rate_down_probability=float(
                    p_down_est
                ),
                rate_reversion_strength=float(
                    strength_est
                ),
                rate_up_probability=float(
                    p_up_est
                ),
                sector_std_dev=float(
                    sector_std_dev
                ),
                theriodic_drift=float(
                    coeff_thr[0]
                ),
                theriodic_idio_std_dev=float(
                    np.std(
                        e_thr
                        - beta_thr_s[0]
                        * sector_shocks,
                        ddof=1,
                    )
                ),
                theriodic_rate_beta=float(
                    coeff_thr[1]
                ),
                theriodic_sector_beta=float(
                    beta_thr_s[0]
                ),
            )

        except Exception:
            # Keep fallback prior parameters if estimation fails.
            pass

    def price_option_from_parameters(
        self,
        market_parameters: MarketParameters,
        option: BinaryOption,
    ) -> float:
        """
        Monte Carlo pricing with exception handling and boundary clipping.
        Improved with higher path count, fixed seed, and removed rounding.
        """
        try:
            curr_values = {
                u.underlying_id: u.value
                for u in self.underlying_state
            }

            steps = option.steps_until_expiry
            if steps == 0:
                return float(option.expiry_valuation(curr_values))

            # Increase paths and fix seed for reproducibility
            num_paths = 10000
            rng = np.random.RandomState(42)  # deterministic without resetting global state

            rates = np.full(num_paths, curr_values[FED_FUNDS_RATE_UNDERLYING_ID])
            ajr = np.full(num_paths, curr_values[AJARAI_UNDERLYING_ID])
            thr = np.full(num_paths, curr_values[THERIODIC_UNDERLYING_ID])

            for _ in range(steps):
                tilts = market_parameters.rate_reversion_strength * (
                    market_parameters.rate_target - rates
                )

                p_up = np.clip(
                    market_parameters.rate_up_probability + tilts,
                    0.0, 1.0
                )
                p_down = np.clip(
                    market_parameters.rate_down_probability - tilts,
                    0.0, 1.0 - p_up
                )
                p_stay = 1.0 - p_up - p_down
                # Ensure p_stay non-negative (should be already due to clip)
                p_stay = np.maximum(0.0, p_stay)

                # Sample rate changes using multinomial probabilities
                rand_draws = rng.rand(num_paths)
                rate_changes = np.zeros(num_paths)

                # Up move
                up_mask = rand_draws < p_up
                rate_changes[up_mask] = market_parameters.rate_step

                # Down move (next segment)
                down_mask = (rand_draws >= p_up) & (rand_draws < p_up + p_down)
                rate_changes[down_mask] = -market_parameters.rate_step

                # Stay move leaves rate_changes = 0

                next_rates = np.maximum(0.0, rates + rate_changes)
                rate_changes = next_rates - rates
                rates = next_rates
                # No rounding on rates (discrete steps already exact)

                # Sector shocks
                sector_shocks = rng.normal(
                    0.0, market_parameters.sector_std_dev, num_paths
                )

                # Stock log-returns (all components included)
                r_ajr = (
                    market_parameters.ajarai_drift
                    + market_parameters.ajarai_rate_beta * rate_changes
                    + market_parameters.ajarai_sector_beta * sector_shocks
                    + rng.normal(
                        0.0, market_parameters.ajarai_idio_std_dev, num_paths
                    )
                )

                r_thr = (
                    market_parameters.theriodic_drift
                    + market_parameters.theriodic_rate_beta * rate_changes
                    + market_parameters.theriodic_sector_beta * sector_shocks
                    + rng.normal(
                        0.0, market_parameters.theriodic_idio_std_dev, num_paths
                    )
                )

                # Update prices without rounding (exact floating)
                ajr = ajr * np.exp(r_ajr)
                thr = thr * np.exp(r_thr)

            # Option payoff aggregation
            obs_values = np.zeros(num_paths)
            for leg in option.legs:
                if leg.underlying_id == FED_FUNDS_RATE_UNDERLYING_ID:
                    obs_values += leg.weight * rates
                elif leg.underlying_id == AJARAI_UNDERLYING_ID:
                    obs_values += leg.weight * ajr
                elif leg.underlying_id == THERIODIC_UNDERLYING_ID:
                    obs_values += leg.weight * thr

            # Binary option price = probability of in-the-money
            price = np.mean(obs_values >= option.strike)
            return float(np.clip(price, 0.0, 1.0))

        except Exception:
            return 0.5

    def price_option(
        self,
        option: BinaryOption,
    ) -> float:
        return self.price_option_from_parameters(
            self.estimated_parameters,
            option,
        )

    def quote(
        self,
        option: BinaryOption,
        counterparty_id: int,
    ) -> Quote:
        """
        Inventory-aware two-sided quoting with conservative risk controls.
        """
        try:
            fair_value = self.price_option(
                option
            )

            curr_inv = (
                self.position
                .option_quantity_by_option_id[
                    option.option_id
                ]
            )

            # ----------------------------------------------------------
            # 1. Mildly tighter dynamic spread.
            #
            # Keep the spread close to the original 0.08 while making
            # liquid-looking binary probabilities slightly more competitive.
            # ----------------------------------------------------------

            probability_uncertainty = (
                fair_value
                * (1.0 - fair_value)
            )

            base_spread = (
                0.050
                + 0.020
                * (
                    4.0
                    * probability_uncertainty
                )
            )

            # Keep spread within a conservative range.
            base_spread = float(
                np.clip(
                    base_spread,
                    0.050,
                    0.070,
                )
            )

            # Multi-leg options have more model uncertainty, so keep
            # their spread slightly wider.
            if len(option.legs) > 1:
                base_spread = min(
                    0.075,
                    base_spread + 0.005,
                )

            # ----------------------------------------------------------
            # 2. Conservative inventory skew.
            #
            # This is stronger than the original -0.003 coefficient,
            # but substantially less aggressive than -0.008.
            # ----------------------------------------------------------

            inventory_skew = (
                -0.0045 * curr_inv
            )

            bid_raw = (
                fair_value
                - base_spread / 2.0
                + inventory_skew
            )

            offer_raw = (
                fair_value
                + base_spread / 2.0
                + inventory_skew
            )

            # ----------------------------------------------------------
            # 3. Round to whole pennies.
            # ----------------------------------------------------------

            bid_price = round(
                min(
                    max(
                        bid_raw,
                        0.01,
                    ),
                    0.98,
                ),
                2,
            )

            offer_price = round(
                max(
                    min(
                        offer_raw,
                        0.99,
                    ),
                    bid_price + 0.01,
                ),
                2,
            )

            # ----------------------------------------------------------
            # 4. Keep the original quote quantity structure.
            # ----------------------------------------------------------

            quote_qty = 2

            # If near the long limit, make the bid unattractive.
            if curr_inv >= self.MAX_INVENTORY:
                bid_price = 0.01

            # If near the short limit, make the offer unattractive.
            elif curr_inv <= -self.MAX_INVENTORY:
                offer_price = 0.99

            # ----------------------------------------------------------
            # 5. Preserve the original cash reservation check.
            # ----------------------------------------------------------

            max_buy_loss = (
                bid_price
                * quote_qty
            )

            max_sell_loss = (
                (1.0 - offer_price)
                * quote_qty
            )

            if (
                max_buy_loss
                + max_sell_loss
                > (
                    self.cash_balance
                    - self.MIN_CASH_BUFFER
                )
            ):
                quote_qty = 1

            return Quote(
                bid_price=bid_price,
                bid_quantity=max(
                    quote_qty,
                    1,
                ),
                offer_price=offer_price,
                offer_quantity=max(
                    quote_qty,
                    1,
                ),
            )

        except Exception:
            # Fail-safe quote to prevent runtime errors.
            return Quote(
                bid_price=0.01,
                bid_quantity=1,
                offer_price=0.99,
                offer_quantity=1,
            )

    def respond_to_fok(
        self,
        option: BinaryOption,
        fok_order: FokOrder,
    ) -> bool:
        """
        Three-layer FOK risk gate:
        inventory limit, cash solvency, and price edge.
        """
        try:
            fair_value = self.price_option(
                option
            )

            curr_inv = (
                self.position
                .option_quantity_by_option_id[
                    option.option_id
                ]
            )

            qty = fok_order.quantity

            # ----------------------------------------------------------
            # 1. Hard inventory limit.
            # ----------------------------------------------------------

            new_inv = (
                curr_inv
                + (
                    -qty
                    if fok_order.order_type
                    == OrderType.BUY
                    else qty
                )
            )

            if abs(new_inv) > self.MAX_INVENTORY:
                return False

            # ----------------------------------------------------------
            # 2. Margin and cash solvency check.
            # ----------------------------------------------------------

            if (
                fok_order.order_type
                == OrderType.BUY
            ):
                # We sell the option.
                max_loss = (
                    1.0
                    - fok_order.price
                ) * qty

            else:
                # We buy the option.
                max_loss = (
                    fok_order.price
                    * qty
                )

            if (
                max_loss
                > (
                    self.cash_balance
                    - self.MIN_CASH_BUFFER
                )
            ):
                return False

            # ----------------------------------------------------------
            # 3. Adaptive safety buffer.
            #
            # Start from the original 0.03 and only reduce it modestly.
            # Inventory imbalance can make the threshold slightly looser
            # when accepting a trade helps flatten the position.
            # ----------------------------------------------------------

            safety_buffer = 0.025

            old_abs_inventory = abs(
                curr_inv
            )

            new_abs_inventory = abs(
                new_inv
            )

            if (
                new_abs_inventory
                < old_abs_inventory
            ):
                safety_buffer = 0.020

            elif (
                new_abs_inventory
                > old_abs_inventory
            ):
                safety_buffer = 0.027

            # ----------------------------------------------------------
            # FOK price test.
            # ----------------------------------------------------------

            if (
                fok_order.order_type
                == OrderType.BUY
            ):
                # The counterparty buys from us.
                # Their bid must be above our fair value.
                return (
                    fok_order.price
                    >= (
                        fair_value
                        + safety_buffer
                    )
                )

            else:
                # The counterparty sells to us.
                # Their ask must be below our fair value.
                return (
                    fok_order.price
                    <= (
                        fair_value
                        - safety_buffer
                    )
                )

        except Exception:
            return False

    def on_step_advance(
        self,
        new_underlying_state: list[Underlying],
        new_option_state: list[BinaryOption],
    ) -> None:

        self.underlying_state = (
            new_underlying_state
        )

        self.active_option_state = (
            new_option_state
        )

    def on_trade(
        self,
        option: BinaryOption,
        price: float,
        quantity: int,
        counterparty_id: int,
    ) -> None:

        self.position.add_option_quantity(
            option.option_id,
            quantity,
        )

        # Positive quantity means we bought the option.
        if quantity > 0:
            self.cash_balance -= (
                price * quantity
            )

        # Negative quantity means we sold the option.
        else:
            self.cash_balance -= (
                (1.0 - price)
                * abs(quantity)
            )
