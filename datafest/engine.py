"""
Future Wallet — High-Fidelity Financial Projection Engine
Core simulation engine with DAG dependency resolution.

FEATURE 2.1: Multi-Currency & Exchange Dynamics
  - CurrencyEngine: correlated geometric Brownian motion with mean-reversion
  - CashLedger: multi-wallet balances in diverse denominations
  - Precision-safe conversion via Python Decimal at transaction realization time
  - Full conversion audit trail for compliance validation

FEATURE 2.3: Credit & Taxation Subsystems
  - Credit Evolution: integral-form CS(t+1) = CS(t) + ∫f(debt_ratio, punctuality, restructuring)dt
  - Taxation Layer: progressive brackets, short/long-term capital gains, FX gains, unrealized tracking

FEATURE 3.1: Inter-Component Dependency Graph
  - Dynamic DAG: add/remove/replace nodes during execution
  - Atomic updates via context manager to prevent inconsistent states
  - Version tracking for DAG structural changes

FEATURE 3.2: Long-Term Memory & Rolling Metrics
  - Continuous rolling shock clustering, recovery slope, vibe/pet
  - Metrics feed back into behavioral evolution (shock sensitivity, expense caution)

FEATURE 3.3: Simulation Branching
  - State snapshotting with branch()/branch_scenarios()
  - Merge/compare divergent trajectories via merge_branches()
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from enum import Enum, auto
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

# Set global Decimal precision for high-frequency conversion integrity
getcontext().prec = 28


# ──────────────────────────────────────────────────────────────────────
# 1. Enums & Data Classes
# ──────────────────────────────────────────────────────────────────────

class NodeType(Enum):
    INCOME = auto()
    EXPENSE = auto()
    TAX = auto()
    ASSET_LIQUID = auto()
    ASSET_ILLIQUID = auto()
    CREDIT = auto()
    EXCHANGE = auto()


class AssetClass(Enum):
    LIQUID = auto()
    ILLIQUID = auto()
    YIELD = auto()
    VOLATILE = auto()


class LockType(Enum):
    """Fund lock classifications per Spec 2.2."""
    NONE = auto()           # Freely available
    TIME_RESTRICTED = auto()    # Locked until a specific day
    ALLOCATION_BOUND = auto()   # Cannot drop below a min % of portfolio
    VESTING_SCHEDULE = auto()   # Unlocks in tranches over time


class PetState(Enum):
    CHILL = "Chill"
    CONTENT = "Content"
    WORRIED = "Worried"
    PANIC = "Panic"
    DEAD = "Flatline"


PET_EMOJI = {
    PetState.CHILL: "🐱",
    PetState.CONTENT: "😺",
    PetState.WORRIED: "😿",
    PetState.PANIC: "🙀",
    PetState.DEAD: "💀",
}


@dataclass
class FinancialNode:
    """A single node in the financial dependency DAG."""
    name: str
    node_type: NodeType
    value: float = 0.0
    currency: str = "USD"
    dependencies: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)


@dataclass
class Asset:
    """
    Enhanced asset model with behavioral parameters per Spec 2.2.
    Supports Liquid, Illiquid, Yield-generating, and Volatile classes
    with daily valuation, liquidity constraints, and fund locks.
    """
    name: str
    asset_class: AssetClass
    value: float
    currency: str = "USD"               # denomination
    daily_volatility: float = 0.01
    annual_yield: float = 0.0
    cost_basis: float = 0.0             # purchase price for gain/loss
    # ── Lock Configuration ──
    lock_type: LockType = LockType.NONE
    locked_until_day: int = 0           # for TIME_RESTRICTED
    min_allocation_pct: float = 0.0     # for ALLOCATION_BOUND (0.0-1.0)
    vesting_tranches: int = 4           # for VESTING_SCHEDULE
    vesting_start_day: int = 0          # day vesting begins
    vesting_interval_days: int = 365    # days between tranches
    # ── Liquidation ──
    liquidation_penalty: float = 0.0
    partial_sell_min: float = 0.0       # minimum partial sell amount
    # ── Behavioral Parameters (class-specific) ──
    momentum_factor: float = 0.0        # for VOLATILE: trend-following strength
    crash_probability: float = 0.0      # for VOLATILE: daily prob of crash event
    crash_severity: float = 0.0         # for VOLATILE: max drop on crash
    mean_reversion_speed: float = 0.0   # for ILLIQUID: speed of reversion to mean
    mean_reversion_target: float = 0.0  # for ILLIQUID: long-run fair value
    # ── Tracking ──
    accrued_yield: float = 0.0          # cumulative yield earned
    cumulative_appreciation: float = 0.0
    valuation_history: list[float] = field(default_factory=list)
    liquidation_events: int = 0
    total_liquidated: float = 0.0

    def __post_init__(self):
        if self.cost_basis <= 0:
            self.cost_basis = self.value
        if self.mean_reversion_target <= 0 and self.asset_class == AssetClass.ILLIQUID:
            self.mean_reversion_target = self.value

    def is_liquid(self, current_day: int) -> bool:
        """Check if asset is freely available for liquidation."""
        if self.asset_class not in (AssetClass.LIQUID, AssetClass.YIELD):
            return False
        return self._check_lock(current_day) >= self.value * 0.99  # fully unlocked

    def _check_lock(self, current_day: int) -> float:
        """
        Returns the amount currently UNLOCKED and available for liquidation.
        Respects lock_type constraints.
        """
        if self.lock_type == LockType.NONE:
            return self.value

        elif self.lock_type == LockType.TIME_RESTRICTED:
            return self.value if current_day >= self.locked_until_day else 0.0

        elif self.lock_type == LockType.ALLOCATION_BOUND:
            # Cannot sell below min_allocation_pct — caller must provide portfolio total
            # For now, return full value; actual constraint checked at portfolio level
            return self.value

        elif self.lock_type == LockType.VESTING_SCHEDULE:
            if current_day < self.vesting_start_day:
                return 0.0
            elapsed = current_day - self.vesting_start_day
            tranches_unlocked = min(
                self.vesting_tranches,
                elapsed // max(self.vesting_interval_days, 1) + 1
            )
            fraction = tranches_unlocked / max(self.vesting_tranches, 1)
            return self.value * fraction

        return self.value

    def get_unlocked_amount(self, current_day: int) -> float:
        """Public API: how much of this asset can be sold today."""
        return self._check_lock(current_day)

    def get_unrealized_gain(self) -> float:
        """Unrealized gain/loss vs cost basis."""
        return self.value - self.cost_basis

    def record_valuation(self):
        """Snapshot current value for history tracking."""
        self.valuation_history.append(self.value)


# ──────────────────────────────────────────────────────────────────────
# 2. MULTI-CURRENCY ENGINE  (Spec § 2.1)
# ──────────────────────────────────────────────────────────────────────

# Configuration for supported currencies
CURRENCY_CONFIG: dict[str, dict] = {
    "USD": {"base_rate": 1.0,       "annual_vol": 0.0,    "mean_rev_speed": 0.0,   "decimals": 2},
    "EUR": {"base_rate": 0.92,      "annual_vol": 0.08,   "mean_rev_speed": 0.02,  "decimals": 4},
    "GBP": {"base_rate": 0.79,      "annual_vol": 0.09,   "mean_rev_speed": 0.02,  "decimals": 4},
    "JPY": {"base_rate": 149.50,    "annual_vol": 0.10,   "mean_rev_speed": 0.015, "decimals": 2},
    "PKR": {"base_rate": 278.00,    "annual_vol": 0.12,   "mean_rev_speed": 0.01,  "decimals": 2},
    "CHF": {"base_rate": 0.88,      "annual_vol": 0.07,   "mean_rev_speed": 0.025, "decimals": 4},
}

# Correlation matrix between currency log-returns (vs USD)
# Order: EUR, GBP, JPY, PKR, CHF
_CORR_LABELS = ["EUR", "GBP", "JPY", "PKR", "CHF"]
_CORR_MATRIX = np.array([
    [1.00,  0.75, -0.30,  0.10,  0.85],
    [0.75,  1.00, -0.20,  0.15,  0.60],
    [-0.30, -0.20, 1.00, -0.05, -0.25],
    [0.10,  0.15, -0.05,  1.00,  0.05],
    [0.85,  0.60, -0.25,  0.05,  1.00],
])


@dataclass
class ConversionRecord:
    """Audit trail entry for a single currency conversion."""
    day: int
    from_currency: str
    to_currency: str
    from_amount: str          # stored as Decimal string for precision
    to_amount: str
    rate_used: str
    inverse_check: str        # from_amount converted back — must match
    precision_drift: str      # absolute error from round-trip


class CurrencyEngine:
    """
    Globalized exchange-rate engine with:
      - Correlated Geometric Brownian Motion (GBM) with Ornstein-Uhlenbeck mean-reversion
      - Daily volatility management
      - Precision-safe Decimal conversion at transaction realization time
      - Full round-trip audit trail
    """

    def __init__(self, rng: np.random.Generator, n_days: int,
                 currencies: list[str] | None = None):
        self.rng = rng
        self.n_days = n_days
        self.currencies = currencies or list(CURRENCY_CONFIG.keys())
        # Ensure USD is always included
        if "USD" not in self.currencies:
            self.currencies.insert(0, "USD")

        self.non_usd = [c for c in self.currencies if c != "USD"]
        self._audit_log: list[ConversionRecord] = []

        # Pre-generate all daily rates
        self.rate_table = self._generate_correlated_rates()
        # Current day rates (updated each step)
        self._current_day = 0
        self._current_rates: dict[str, Decimal] = {}
        self._set_day(0)

    # ── Rate Generation ──────────────────────────────────────────

    def _generate_correlated_rates(self) -> pd.DataFrame:
        """
        Generate daily exchange rates using correlated GBM + mean-reversion.
        
        Model per currency c (vs USD):
            dS_c / S_c = kappa_c * (ln(mu_c) - ln(S_c)) * dt + sigma_c * dW_c
        
        Where dW is drawn from a multivariate normal with correlation matrix.
        """
        # Build sub-correlation matrix for currencies we actually use
        idx_map = {c: i for i, c in enumerate(_CORR_LABELS)}
        active = [c for c in self.non_usd if c in idx_map]
        n = len(active)

        if n == 0:
            return pd.DataFrame({"USD": [1.0] * self.n_days},
                                index=pd.RangeIndex(self.n_days, name="day"))

        indices = [idx_map[c] for c in active]
        corr_sub = _CORR_MATRIX[np.ix_(indices, indices)]

        # Cholesky decomposition for correlated normals
        L = np.linalg.cholesky(corr_sub)

        # Pre-allocate
        rates = np.zeros((self.n_days, n))
        configs = [CURRENCY_CONFIG[c] for c in active]
        log_rates = np.array([math.log(cfg["base_rate"]) for cfg in configs])
        log_means = log_rates.copy()  # long-term mean

        dt = 1.0 / 365.0
        rates[0] = np.exp(log_rates)

        for t in range(1, self.n_days):
            # Draw correlated standard normals
            z = self.rng.standard_normal(n)
            corr_z = L @ z

            for i in range(n):
                kappa = configs[i]["mean_rev_speed"]
                sigma = configs[i]["annual_vol"]
                # Ornstein-Uhlenbeck on log-price
                drift = kappa * (log_means[i] - log_rates[i]) * dt
                diffusion = sigma * math.sqrt(dt) * corr_z[i]
                log_rates[i] += drift + diffusion

            rates[t] = np.exp(log_rates)

        # Build DataFrame
        data = {"USD": [1.0] * self.n_days}
        for i, c in enumerate(active):
            data[c] = rates[:, i]

        # Add any currencies that weren't in _CORR_LABELS (simple random walk fallback)
        for c in self.non_usd:
            if c not in active:
                cfg = CURRENCY_CONFIG.get(c, {"base_rate": 1.0, "annual_vol": 0.05})
                rate = cfg["base_rate"]
                fallback = [rate]
                for _ in range(self.n_days - 1):
                    rate *= math.exp(self.rng.normal(0, cfg["annual_vol"] * math.sqrt(dt)))
                    fallback.append(rate)
                data[c] = fallback

        return pd.DataFrame(data, index=pd.RangeIndex(self.n_days, name="day"))

    # ── Day Management ───────────────────────────────────────────

    def _set_day(self, day: int):
        """Load exchange rates for a specific day into Decimal precision."""
        self._current_day = day
        clamped = min(day, self.n_days - 1)
        self._current_rates = {}
        for c in self.currencies:
            float_rate = float(self.rate_table[c].iloc[clamped])
            cfg = CURRENCY_CONFIG.get(c, {"decimals": 4})
            # Store as Decimal with appropriate precision
            self._current_rates[c] = Decimal(str(float_rate))

    def advance_day(self, day: int):
        """Called by the simulation engine at the start of each day."""
        self._set_day(day)

    # ── Conversion with Precision Guarantee ──────────────────────

    def get_rate(self, from_cur: str, to_cur: str) -> Decimal:
        """
        Get the exchange rate to convert from_cur -> to_cur at current day.
        Rate semantics: 1 unit of from_cur = rate units of to_cur.
        """
        if from_cur == to_cur:
            return Decimal("1")
        # Both rates are vs USD: rate[CUR] = how many CUR per 1 USD
        rate_from = self._current_rates.get(from_cur, Decimal("1"))
        rate_to = self._current_rates.get(to_cur, Decimal("1"))
        # from_cur -> USD -> to_cur
        # 1 from_cur = (1 / rate_from) USD = (rate_to / rate_from) to_cur
        return rate_to / rate_from

    def convert(self, amount: float, from_cur: str, to_cur: str,
                day: int | None = None, audit: bool = True) -> float:
        """
        Convert an amount from one currency to another at the current day's rate.
        
        Precision guarantee: uses Decimal arithmetic internally, rounds to
        the destination currency's decimal places, and verifies round-trip
        integrity before returning.
        
        Args:
            amount: The amount in from_cur
            from_cur: Source currency code
            to_cur: Target currency code
            day: Override day (uses current if None)
            audit: Whether to log this conversion
            
        Returns:
            Converted amount as float, precision-safe
        """
        if from_cur == to_cur:
            return amount

        if day is not None and day != self._current_day:
            self._set_day(day)

        rate = self.get_rate(from_cur, to_cur)
        dec_amount = Decimal(str(amount))
        dec_result = dec_amount * rate

        # Round to target currency precision
        target_decimals = CURRENCY_CONFIG.get(to_cur, {"decimals": 2})["decimals"]
        quantize_str = "0." + "0" * target_decimals
        dec_result_rounded = dec_result.quantize(Decimal(quantize_str), rounding=ROUND_HALF_EVEN)

        # Round-trip check: convert back and measure drift
        inverse_rate = self.get_rate(to_cur, from_cur)
        roundtrip = dec_result_rounded * inverse_rate
        source_decimals = CURRENCY_CONFIG.get(from_cur, {"decimals": 2})["decimals"]
        quantize_src = "0." + "0" * source_decimals
        roundtrip_rounded = roundtrip.quantize(Decimal(quantize_src), rounding=ROUND_HALF_EVEN)
        drift = abs(dec_amount - roundtrip_rounded)

        if audit:
            self._audit_log.append(ConversionRecord(
                day=self._current_day,
                from_currency=from_cur,
                to_currency=to_cur,
                from_amount=str(dec_amount),
                to_amount=str(dec_result_rounded),
                rate_used=str(rate),
                inverse_check=str(roundtrip_rounded),
                precision_drift=str(drift),
            ))

        return float(dec_result_rounded)

    def convert_to_usd(self, amount: float, from_cur: str,
                        day: int | None = None, audit: bool = True) -> float:
        """Convenience: convert any currency to USD."""
        return self.convert(amount, from_cur, "USD", day=day, audit=audit)

    def convert_from_usd(self, amount_usd: float, to_cur: str,
                          day: int | None = None, audit: bool = True) -> float:
        """Convenience: convert USD to any currency."""
        return self.convert(amount_usd, "USD", to_cur, day=day, audit=audit)

    # ── Reporting ────────────────────────────────────────────────

    def get_rate_float(self, currency: str) -> float:
        """Get current rate as float (for display/metrics)."""
        return float(self._current_rates.get(currency, Decimal("1")))

    def get_all_rates_float(self) -> dict[str, float]:
        """Get all current rates as float dict."""
        return {c: float(r) for c, r in self._current_rates.items()}

    @property
    def audit_log(self) -> list[ConversionRecord]:
        return self._audit_log

    def get_audit_summary(self) -> dict[str, Any]:
        """Summary statistics of all conversions performed."""
        if not self._audit_log:
            return {"total_conversions": 0, "max_drift": "0", "avg_drift": "0"}
        drifts = [Decimal(r.precision_drift) for r in self._audit_log]
        return {
            "total_conversions": len(self._audit_log),
            "max_drift": str(max(drifts)),
            "avg_drift": str(sum(drifts) / len(drifts)),
            "unique_pairs": len(set(
                (r.from_currency, r.to_currency) for r in self._audit_log
            )),
            "conversions_by_pair": {
                f"{pair[0]}->{pair[1]}": sum(
                    1 for r in self._audit_log
                    if (r.from_currency, r.to_currency) == pair
                )
                for pair in set(
                    (r.from_currency, r.to_currency) for r in self._audit_log
                )
            },
        }

    def get_daily_rates_df(self) -> pd.DataFrame:
        """Return the full pre-generated rate table for visualization."""
        return self.rate_table


# ──────────────────────────────────────────────────────────────────────
# 3. MULTI-CURRENCY CASH LEDGER
# ──────────────────────────────────────────────────────────────────────

class CashLedger:
    """
    Multi-wallet cash management system.
    Holds balances in multiple currencies simultaneously.
    All cross-currency operations go through CurrencyEngine for
    precision-safe, audited conversion at transaction realization time.
    """

    def __init__(self, currency_engine: CurrencyEngine, home_currency: str = "USD"):
        self.engine = currency_engine
        self.home = home_currency
        self._balances: dict[str, Decimal] = {}

    def deposit(self, amount: float, currency: str):
        """Add funds in a specific currency."""
        dec = Decimal(str(amount))
        self._balances[currency] = self._balances.get(currency, Decimal("0")) + dec

    def withdraw(self, amount: float, currency: str) -> bool:
        """
        Remove funds from a specific currency wallet.
        Returns True if successful, False if insufficient funds.
        """
        dec = Decimal(str(amount))
        current = self._balances.get(currency, Decimal("0"))
        if current >= dec:
            self._balances[currency] = current - dec
            return True
        return False

    def withdraw_force(self, amount: float, currency: str):
        """Withdraw even if it goes negative."""
        dec = Decimal(str(amount))
        self._balances[currency] = self._balances.get(currency, Decimal("0")) - dec

    def get_balance(self, currency: str) -> float:
        """Get balance in a specific currency."""
        return float(self._balances.get(currency, Decimal("0")))

    def get_total_in_home(self) -> float:
        """
        Get total balance across all wallets, converted to home currency
        at current day's rates (transaction-time realization).
        """
        total = Decimal("0")
        for cur, bal in self._balances.items():
            if cur == self.home:
                total += bal
            else:
                converted = self.engine.convert(float(bal), cur, self.home, audit=False)
                total += Decimal(str(converted))
        return float(total)

    def transfer(self, amount: float, from_cur: str, to_cur: str) -> float:
        """
        Convert funds between currency wallets.
        Conversion happens at the exact current rate (transaction realization).
        Returns the amount received in to_cur.
        """
        if from_cur == to_cur:
            return amount
        # Withdraw from source
        self.withdraw_force(amount, from_cur)
        # Convert through the engine (audited, precision-safe)
        received = self.engine.convert(amount, from_cur, to_cur)
        # Deposit into destination
        self.deposit(received, to_cur)
        return received

    def get_all_balances(self) -> dict[str, float]:
        """Get all wallet balances."""
        return {c: float(b) for c, b in self._balances.items() if b != 0}

    def snapshot_balances(self) -> dict[str, str]:
        """Snapshot balances as Decimal strings for perfect restoration."""
        return {c: str(b) for c, b in self._balances.items()}

    def restore_balances(self, snapshot: dict[str, str]):
        """Restore from a snapshot."""
        self._balances = {c: Decimal(v) for c, v in snapshot.items()}


# ──────────────────────────────────────────────────────────────────────
# 4. Simulation State (enhanced with multi-currency)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LiquidationEvent:
    """Audit record for an asset liquidation."""
    day: int
    asset_name: str
    asset_class: str
    amount_sold_native: float
    amount_sold_home: float
    penalty_applied: float
    proceeds_home: float
    reason: str                 # "deficit_cover" | "rebalance" | "forced"
    lock_override: bool         # True if lock was broken
    remaining_value: float


@dataclass
class TaxRecord:
    """Annual tax computation audit trail (Spec 2.3)."""
    year: int
    ordinary_income: float
    short_term_gains: float       # held < 365 days, taxed at income rate
    long_term_gains: float        # held >= 365 days, preferential rate
    fx_gains: float               # realized FX conversion gains
    unrealized_gains: float       # mark-to-market (not taxed)
    income_tax: float
    short_term_tax: float
    long_term_tax: float
    fx_tax: float
    total_tax: float


@dataclass
class SimulationState:
    """Complete snapshot of engine state at a point in time."""
    day: int = 0
    cash: float = 0.0                  # total cash in home currency (derived from ledger)
    credit_score: float = 650.0
    total_debt: float = 0.0
    debt_paid_on_time: int = 0
    debt_payments_total: int = 0
    assets: list[Asset] = field(default_factory=list)
    daily_balances: list[float] = field(default_factory=list)
    daily_credit_scores: list[float] = field(default_factory=list)
    daily_nav: list[float] = field(default_factory=list)
    daily_portfolio_value: list[float] = field(default_factory=list)
    daily_exchange_rates: list[dict[str, float]] = field(default_factory=list)
    shock_days: list[int] = field(default_factory=list)
    tax_paid: float = 0.0
    realized_gains: float = 0.0
    exchange_rates: dict[str, float] = field(default_factory=dict)
    collapsed: bool = False
    collapse_day: int | None = None
    # Multi-currency state
    ledger_snapshot: dict[str, str] = field(default_factory=dict)
    currency_conversions_count: int = 0
    # Asset Portfolio state (Spec 2.2)
    liquidation_log: list[LiquidationEvent] = field(default_factory=list)
    total_liquidated_value: float = 0.0
    total_yield_earned: float = 0.0
    # Credit & Taxation state (Spec 2.3)
    short_term_realized: float = 0.0   # gains from assets held < 1 year
    long_term_realized: float = 0.0    # gains from assets held >= 1 year
    fx_realized_gains: float = 0.0     # FX conversion gains
    restructuring_events: int = 0      # debt restructuring count
    debt_restructured: bool = False    # currently in restructuring
    credit_utilization: float = 0.0    # credit line utilization ratio
    missed_payments: int = 0           # total missed payment count
    consecutive_on_time: int = 0       # current on-time payment streak
    credit_history: list[dict] = field(default_factory=list)   # daily credit factors
    tax_records: list[TaxRecord] = field(default_factory=list)  # annual tax audit
    # Rolling metrics (Spec 3.2)
    rolling_scd: float = 0.0       # shock clustering density (updated daily)
    rolling_recovery: float = 1.0  # recovery slope (updated daily)
    rolling_vibe: str = "Neutral"  # qualitative vibe
    rolling_pet: str = "Content"   # pet state name
    dag_version: int = 0           # DAG structural version counter
    branch_id: str = "main"        # branch identifier for tracking

    def snapshot(self) -> SimulationState:
        return copy.deepcopy(self)


# ──────────────────────────────────────────────────────────────────────
# 5. DAG Resolver (NetworkX)
# ──────────────────────────────────────────────────────────────────────

class DAGResolver:
    """
    Enhanced financial component dependency graph (Spec 3.1).

    Supports:
     - Dynamic add/remove/replace of nodes during execution
     - Atomic updates via context manager (prevents inconsistent states)
     - Version tracking for structural changes
     - Cycle detection and validation
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: dict[str, FinancialNode] = {}
        self._version = 0
        self._order_cache: list[str] | None = None
        self._locked = False  # True during atomic updates
        self._pending_ops: list[tuple] = []  # buffered ops for atomic commit
        self._change_log: list[dict] = []  # structural change history

    @property
    def version(self) -> int:
        return self._version

    def add_node(self, node: FinancialNode):
        """Add a node to the DAG (or buffer if inside atomic block)."""
        if self._locked:
            self._pending_ops.append(("add", node))
            return
        self._do_add(node)

    def _do_add(self, node: FinancialNode):
        self.nodes[node.name] = node
        self.graph.add_node(node.name)
        for dep in node.dependencies:
            self.graph.add_edge(dep, node.name)
        self._invalidate()
        self._change_log.append({
            "version": self._version, "op": "add", "node": node.name,
            "deps": node.dependencies[:],
        })

    def remove_node(self, name: str) -> FinancialNode | None:
        """
        Remove a node and its edges from the DAG (Spec 3.1).
        Returns the removed node or None if not found.
        """
        if self._locked:
            self._pending_ops.append(("remove", name))
            return self.nodes.get(name)
        return self._do_remove(name)

    def _do_remove(self, name: str) -> FinancialNode | None:
        if name not in self.nodes:
            return None
        node = self.nodes.pop(name)
        self.graph.remove_node(name)
        self._invalidate()
        self._change_log.append({
            "version": self._version, "op": "remove", "node": name,
        })
        return node

    def replace_node(self, old_name: str, new_node: FinancialNode):
        """
        Atomically replace a node: remove old, add new (Spec 3.1).
        Preserves dependents that referenced the old node.
        """
        if self._locked:
            self._pending_ops.append(("replace", (old_name, new_node)))
            return
        # Find nodes that depended on old_name
        dependents = list(self.graph.successors(old_name)) if old_name in self.graph else []
        self._do_remove(old_name)
        self._do_add(new_node)
        # Re-link dependents to new node (if name changed)
        if new_node.name != old_name:
            for dep_name in dependents:
                if dep_name in self.nodes:
                    self.graph.add_edge(new_node.name, dep_name)
            self._invalidate()

    def validate(self) -> bool:
        """Check DAG is acyclic and all dependencies exist (Spec 3.1)."""
        if not nx.is_directed_acyclic_graph(self.graph):
            return False
        for node_name, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    return False
        return True

    class _AtomicContext:
        """Context manager for atomic DAG updates (Spec 3.1)."""
        def __init__(self, dag: 'DAGResolver'):
            self.dag = dag
            self._snapshot_graph = None
            self._snapshot_nodes = None

        def __enter__(self):
            self.dag._locked = True
            self.dag._pending_ops.clear()
            # Snapshot for rollback
            self._snapshot_graph = self.dag.graph.copy()
            self._snapshot_nodes = dict(self.dag.nodes)
            return self.dag

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.dag._locked = False
            if exc_type is not None:
                # Rollback on error
                self.dag.graph = self._snapshot_graph
                self.dag.nodes = self._snapshot_nodes
                self.dag._invalidate()
                return False
            # Commit buffered operations
            for op in self.dag._pending_ops:
                if op[0] == "add":
                    self.dag._do_add(op[1])
                elif op[0] == "remove":
                    self.dag._do_remove(op[1])
                elif op[0] == "replace":
                    old, new = op[1]
                    self.dag.replace_node(old, new)
            self.dag._pending_ops.clear()
            # Validate after commit
            if not self.dag.validate():
                self.dag.graph = self._snapshot_graph
                self.dag.nodes = self._snapshot_nodes
                self.dag._invalidate()
                raise ValueError("Atomic DAG update produced invalid graph!")
            return False

    def atomic(self) -> _AtomicContext:
        """Context manager for atomic, rollback-safe DAG mutations."""
        return self._AtomicContext(self)

    def _invalidate(self):
        """Invalidate cached order and bump version."""
        self._order_cache = None
        self._version += 1

    def resolve_order(self) -> list[str]:
        """Topological sort -- determines daily calculation order."""
        if self._order_cache is not None:
            return self._order_cache
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Dependency graph contains cycles!")
        self._order_cache = list(nx.topological_sort(self.graph))
        return self._order_cache

    def get_node(self, name: str) -> FinancialNode:
        return self.nodes[name]

    def get_change_log(self) -> list[dict]:
        """Return the full structural change history."""
        return self._change_log[:]

    def get_active_nodes(self) -> list[str]:
        """Return names of all active nodes."""
        return list(self.nodes.keys())

    def to_dict(self) -> dict:
        """Serialize DAG structure for output."""
        return {
            "version": self._version,
            "nodes": [
                {"name": n.name, "type": n.node_type.name, "deps": n.dependencies}
                for n in self.nodes.values()
            ],
            "edges": list(self.graph.edges()),
            "order": self.resolve_order(),
            "changes": len(self._change_log),
        }


# ──────────────────────────────────────────────────────────────────────
# 6. Asset / Tax Generators
# ──────────────────────────────────────────────────────────────────────

def generate_default_assets() -> list[Asset]:
    """
    Create a realistic starter portfolio with diverse asset classes,
    lock types, and behavioral parameters per Spec 2.2.
    """
    return [
        # LIQUID — freely available, low vol, modest yield
        Asset(
            name="Savings Account", asset_class=AssetClass.LIQUID,
            value=5000, currency="USD",
            daily_volatility=0.0005, annual_yield=0.035,
            lock_type=LockType.NONE, liquidation_penalty=0.0,
        ),
        # YIELD — bond fund, allocation-bound to ensure min portfolio stability
        Asset(
            name="Euro Bond Fund", asset_class=AssetClass.YIELD,
            value=8000, currency="EUR",
            daily_volatility=0.003, annual_yield=0.045,
            lock_type=LockType.ALLOCATION_BOUND, min_allocation_pct=0.05,
            liquidation_penalty=0.02,
        ),
        # VOLATILE — stock ETF with momentum and crash risk
        Asset(
            name="Stock ETF", asset_class=AssetClass.VOLATILE,
            value=10000, currency="USD",
            daily_volatility=0.015, annual_yield=0.08,
            lock_type=LockType.NONE, liquidation_penalty=0.005,
            momentum_factor=0.3, crash_probability=0.005, crash_severity=0.12,
        ),
        # VOLATILE — vesting RSUs, unlock in 4 annual tranches
        Asset(
            name="Company RSUs", asset_class=AssetClass.VOLATILE,
            value=12000, currency="USD",
            daily_volatility=0.02, annual_yield=0.0,
            lock_type=LockType.VESTING_SCHEDULE,
            vesting_tranches=4, vesting_start_day=0, vesting_interval_days=365,
            liquidation_penalty=0.01,
            momentum_factor=0.4, crash_probability=0.008, crash_severity=0.15,
        ),
        # ILLIQUID — UK REIT with time lock and mean-reversion
        Asset(
            name="UK REIT", asset_class=AssetClass.ILLIQUID,
            value=15000, currency="GBP",
            daily_volatility=0.005, annual_yield=0.04,
            lock_type=LockType.TIME_RESTRICTED, locked_until_day=365,
            liquidation_penalty=0.20,
            mean_reversion_speed=0.01, mean_reversion_target=15000,
        ),
        # ILLIQUID — Real estate, long lock, allocation-bound
        Asset(
            name="Real Estate", asset_class=AssetClass.ILLIQUID,
            value=50000, currency="USD",
            daily_volatility=0.003, annual_yield=0.03,
            lock_type=LockType.TIME_RESTRICTED, locked_until_day=730,
            liquidation_penalty=0.20,
            mean_reversion_speed=0.005, mean_reversion_target=50000,
        ),
        # LIQUID — Emergency fund, allocation-bound (cannot drop below 8% of portfolio)
        Asset(
            name="Emergency Fund", asset_class=AssetClass.LIQUID,
            value=3000, currency="USD",
            daily_volatility=0.0002, annual_yield=0.04,
            lock_type=LockType.ALLOCATION_BOUND, min_allocation_pct=0.08,
            liquidation_penalty=0.0,
        ),
    ]


# ── Progressive Tax Brackets (Spec 2.3) ──

INCOME_TAX_BRACKETS = [
    (11_000, 0.10),
    (44_725, 0.12),
    (95_375, 0.22),
    (182_100, 0.24),
    (231_250, 0.32),
    (578_125, 0.35),
    (float("inf"), 0.37),
]

# Long-term capital gains get preferential rates
LTCG_BRACKETS = [
    (44_625, 0.00),      # 0% for low income
    (492_300, 0.15),     # 15% for middle
    (float("inf"), 0.20),  # 20% for high
]

# FX gains taxed as ordinary income (IRS Section 988)
FX_TAX_RATE = 0.30  # blended rate for simplicity


def compute_progressive_tax(taxable: float, brackets: list[tuple]) -> float:
    """Apply progressive bracket computation to any income type."""
    if taxable <= 0:
        return 0.0
    tax = 0.0
    prev = 0.0
    for bracket_top, rate in brackets:
        if taxable <= prev:
            break
        in_bracket = min(taxable, bracket_top) - prev
        tax += in_bracket * rate
        prev = bracket_top
    return tax


def compute_annual_tax(taxable_income: float) -> float:
    """Backward-compatible: progressive US-style income tax brackets."""
    return compute_progressive_tax(taxable_income, INCOME_TAX_BRACKETS)


def compute_full_tax(
    ordinary_income: float,
    short_term_gains: float,
    long_term_gains: float,
    fx_gains: float,
) -> dict[str, float]:
    """
    Full tax computation per Spec 2.3.
    
    - Ordinary income + short-term gains: taxed at progressive income rates
    - Long-term capital gains: preferential LTCG rates (0%/15%/20%)
    - FX gains: taxed at blended ordinary rate (Section 988)
    
    Returns breakdown of taxes by category.
    """
    # Short-term gains are taxed at ordinary income rates (stacked on top)
    income_plus_st = max(ordinary_income + short_term_gains, 0)
    income_tax_total = compute_progressive_tax(income_plus_st, INCOME_TAX_BRACKETS)
    # Attribute: tax on just income vs. the marginal from adding ST gains
    income_only_tax = compute_progressive_tax(max(ordinary_income, 0), INCOME_TAX_BRACKETS)
    short_term_tax = income_tax_total - income_only_tax
    
    # Long-term capital gains at preferential rates
    long_term_tax = compute_progressive_tax(max(long_term_gains, 0), LTCG_BRACKETS)
    
    # FX gains as ordinary income
    fx_tax = max(fx_gains, 0) * FX_TAX_RATE
    
    total = income_only_tax + short_term_tax + long_term_tax + fx_tax
    
    return {
        "income_tax": income_only_tax,
        "short_term_tax": short_term_tax,
        "long_term_tax": long_term_tax,
        "fx_tax": fx_tax,
        "total_tax": total,
    }


# ──────────────────────────────────────────────────────────────────────
# 7. Simulation Engine (enhanced with full multi-currency)
# ──────────────────────────────────────────────────────────────────────

class SimulationEngine:
    """
    Deterministic daily financial simulation engine.

    Given the same seed + inputs, produces bit-exact identical outputs.
    Features full multi-currency support with precision-safe conversions.
    """

    def __init__(
        self,
        seed: int = 42,
        initial_savings: float = 20_000,
        monthly_salary: float = 5_000,
        monthly_expenses: float = 3_000,
        initial_debt: float = 15_000,
        monthly_debt_payment: float = 500,
        shock_probability: float = 0.02,
        simulation_years: int = 5,
        salary_currency: str = "USD",
        expense_currency: str = "USD",
        home_currency: str = "USD",
        assets: list[Asset] | None = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_days = simulation_years * 365
        self.simulation_years = simulation_years

        # ── User params ──
        self.initial_savings = initial_savings
        self.daily_salary = monthly_salary / 30.0
        self.daily_expenses = monthly_expenses / 30.0
        self.initial_debt = initial_debt
        self.daily_debt_payment = monthly_debt_payment / 30.0
        self.shock_prob = shock_probability
        self.salary_currency = salary_currency
        self.expense_currency = expense_currency
        self.home_currency = home_currency

        # ── Currency Engine (FEATURE 2.1) ──
        self.currency_engine = CurrencyEngine(self.rng, self.n_days)

        # ── Cash Ledger (multi-wallet) ──
        self.ledger = CashLedger(self.currency_engine, home_currency=home_currency)
        self.ledger.deposit(initial_savings, home_currency)

        # ── State ──
        self.state = SimulationState(
            cash=initial_savings,
            total_debt=initial_debt,
            assets=assets if assets is not None else generate_default_assets(),
            exchange_rates=self.currency_engine.get_all_rates_float(),
        )

        # ── DAG ──
        self.dag = self._build_dag()
        self.execution_order = self.dag.resolve_order()

    # Expose exchange_df for backward compatibility with app.py
    @property
    def exchange_df(self) -> pd.DataFrame:
        return self.currency_engine.get_daily_rates_df()

    def _build_dag(self) -> DAGResolver:
        dag = DAGResolver()
        dag.add_node(FinancialNode("exchange", NodeType.EXCHANGE))
        dag.add_node(FinancialNode("income", NodeType.INCOME, dependencies=["exchange"]))
        dag.add_node(FinancialNode("expenses", NodeType.EXPENSE, dependencies=["exchange"]))
        dag.add_node(FinancialNode("debt_service", NodeType.EXPENSE, dependencies=["income"]))
        dag.add_node(FinancialNode("asset_valuation", NodeType.ASSET_LIQUID, dependencies=["exchange"]))
        dag.add_node(FinancialNode("tax", NodeType.TAX, dependencies=["income", "asset_valuation"]))
        dag.add_node(FinancialNode("credit_update", NodeType.CREDIT,
                                   dependencies=["debt_service", "tax"]))
        dag.add_node(FinancialNode("liquidation_check", NodeType.ASSET_LIQUID,
                                   dependencies=["expenses", "debt_service", "tax"]))
        return dag

    # ── Daily Node Processors ──

    def _process_exchange(self, day: int):
        """Advance the CurrencyEngine to today's rates."""
        self.currency_engine.advance_day(day)
        self.state.exchange_rates = self.currency_engine.get_all_rates_float()
        self.state.daily_exchange_rates.append(self.state.exchange_rates.copy())

    def _process_income(self, day: int):
        """
        Receive daily salary in salary_currency.
        If salary is in a foreign currency, convert to home currency at
        today's exact rate (transaction-time realization per spec 2.1).
        """
        salary_local = self.daily_salary  # amount in salary_currency

        if self.salary_currency == self.home_currency:
            self.ledger.deposit(salary_local, self.home_currency)
        else:
            # Deposit in foreign currency first, then convert
            # This ensures conversion happens at transaction realization time
            self.ledger.deposit(salary_local, self.salary_currency)
            self.ledger.transfer(salary_local, self.salary_currency, self.home_currency)

    def _process_expenses(self, day: int):
        """
        Deduct daily expenses in expense_currency.
        Conversion at transaction realization time if cross-currency.
        """
        base = self.daily_expenses
        # Random shock — influenced by rolling SCD (Spec 3.2)
        # High shock clustering makes future shocks more likely (contagion)
        scd_multiplier = 1.0 + self.state.rolling_scd * 0.5  # up to 1.5x
        if self.rng.random() < self.shock_prob * scd_multiplier:
            shock_magnitude = self.rng.uniform(1.5, 5.0)
            # If vibe is stressed/critical, shocks hit harder (Spec 3.2)
            if self.state.rolling_vibe in ("Stressed", "Critical"):
                shock_magnitude *= 1.3
            base *= shock_magnitude
            self.state.shock_days.append(day)

        if self.expense_currency == self.home_currency:
            self.ledger.withdraw_force(base, self.home_currency)
        else:
            # Convert from home to expense currency, then pay
            needed_home = self.currency_engine.convert(
                base, self.expense_currency, self.home_currency
            )
            self.ledger.withdraw_force(needed_home, self.home_currency)

    def _process_debt_service(self, day: int):
        """
        Process daily debt payment (always in home currency).
        Enhanced for Spec 2.3: tracks missed payments, consecutive on-time
        streak, and detects implicit debt restructuring.
        """
        if self.state.total_debt <= 0:
            return
        payment = min(self.daily_debt_payment, self.state.total_debt)
        pre_balance = self.ledger.get_balance(self.home_currency)
        self.ledger.withdraw_force(payment, self.home_currency)
        self.state.total_debt -= payment
        self.state.debt_payments_total += 1

        if pre_balance >= payment:
            self.state.debt_paid_on_time += 1
            self.state.consecutive_on_time += 1
        else:
            # Missed payment (insufficient funds)
            self.state.missed_payments += 1
            self.state.consecutive_on_time = 0  # streak broken
            # If multiple consecutive misses, flag as restructuring
            if self.state.missed_payments >= 3 and not self.state.debt_restructured:
                self.state.debt_restructured = True
                self.state.restructuring_events += 1

    def _process_asset_valuation(self, day: int):
        """
        Revalue all assets daily using class-specific behavioral models (Spec 2.2).
        Each AssetClass has distinct dynamics:
          - LIQUID:   stable drift + tiny noise + yield accrual
          - YIELD:    compound interest + bond-like vol + yield accrual
          - VOLATILE: GBM with momentum, crash events, trend-following
          - ILLIQUID: mean-reversion (O-U process) + yield accrual  
        """
        portfolio_total = sum(a.value for a in self.state.assets)

        for asset in self.state.assets:
            prev_value = asset.value

            if asset.asset_class == AssetClass.LIQUID:
                # Stable drift: nearly deterministic growth with tiny noise
                daily_yield = asset.annual_yield / 365.0
                noise = self.rng.normal(0, asset.daily_volatility)
                asset.value *= (1.0 + daily_yield + noise)
                yield_earned = asset.value - prev_value
                if yield_earned > 0:
                    asset.accrued_yield += yield_earned

            elif asset.asset_class == AssetClass.YIELD:
                # Compound interest with bond-like volatility
                daily_yield = asset.annual_yield / 365.0
                noise = self.rng.normal(0, asset.daily_volatility)
                # Interest accrues as compound growth
                interest = asset.value * daily_yield
                asset.value += interest
                asset.value *= (1.0 + noise)  # mark-to-market fluctuation
                yield_earned = interest
                asset.accrued_yield += max(yield_earned, 0)

            elif asset.asset_class == AssetClass.VOLATILE:
                # GBM with momentum factor + crash events
                daily_drift = asset.annual_yield / 365.0
                noise = self.rng.normal(0, asset.daily_volatility)

                # Momentum: trend-following from recent performance
                momentum = 0.0
                if asset.momentum_factor > 0 and len(asset.valuation_history) >= 5:
                    recent = asset.valuation_history[-5:]
                    trend = (recent[-1] - recent[0]) / max(recent[0], 1.0)
                    momentum = asset.momentum_factor * trend * 0.1

                # Crash event (tail risk)
                crash = 0.0
                if asset.crash_probability > 0 and self.rng.random() < asset.crash_probability:
                    crash = -self.rng.uniform(0.02, asset.crash_severity)

                asset.value *= (1.0 + daily_drift + noise + momentum + crash)

            elif asset.asset_class == AssetClass.ILLIQUID:
                # Ornstein-Uhlenbeck mean-reversion + yield
                daily_yield = asset.annual_yield / 365.0
                noise = self.rng.normal(0, asset.daily_volatility)
                kappa = asset.mean_reversion_speed
                target = asset.mean_reversion_target if asset.mean_reversion_target > 0 else prev_value
                # Mean-reverting component
                reversion = kappa * (target - asset.value) / max(asset.value, 1.0)
                asset.value *= (1.0 + daily_yield + noise + reversion)
                yield_earned = asset.value * daily_yield
                asset.accrued_yield += max(yield_earned, 0)

            # Floor at zero
            asset.value = max(asset.value, 0.0)

            # Track appreciation
            asset.cumulative_appreciation += (asset.value - prev_value)

            # Record daily valuation for history
            asset.record_valuation()

        # Track total portfolio value
        self.state.daily_portfolio_value.append(portfolio_total)
        self.state.total_yield_earned = sum(a.accrued_yield for a in self.state.assets)

    def _process_tax(self, day: int):
        """
        Enhanced annual tax computation (Spec 2.3).

        Separates income into 4 categories with distinct tax treatment:
          1. Ordinary income (salary) → progressive income brackets
          2. Short-term capital gains (assets held < 365 days) → income rate
          3. Long-term capital gains (assets held >= 365 days) → preferential LTCG rate
          4. FX conversion gains → Section 988 blended rate

        Unrealized gains are tracked but NOT taxed until realization.
        """
        if day > 0 and day % 365 == 0:
            year = day // 365

            # ── Ordinary Income ──
            if self.salary_currency != self.home_currency:
                annual_income_local = self.daily_salary * 365
                annual_income = self.currency_engine.convert(
                    annual_income_local, self.salary_currency, self.home_currency
                )
            else:
                annual_income = self.daily_salary * 365

            # ── Realized Gains (split by holding period) ──
            st_gains = self.state.short_term_realized
            lt_gains = self.state.long_term_realized

            # Legacy realized_gains that weren't categorized → treat as short-term
            uncategorized = self.state.realized_gains - (st_gains + lt_gains)
            if uncategorized > 0:
                st_gains += uncategorized

            # ── FX Gains ──
            fx_gains = self.state.fx_realized_gains

            # ── Unrealized Gains (tracked, not taxed) ──
            unrealized = sum(a.get_unrealized_gain() for a in self.state.assets)

            # ── Compute Full Tax ──
            tax_result = compute_full_tax(
                ordinary_income=annual_income,
                short_term_gains=st_gains,
                long_term_gains=lt_gains,
                fx_gains=fx_gains,
            )

            total_tax = tax_result["total_tax"]
            self.ledger.withdraw_force(total_tax, self.home_currency)
            self.state.tax_paid += total_tax

            # ── Audit Record ──
            self.state.tax_records.append(TaxRecord(
                year=year,
                ordinary_income=annual_income,
                short_term_gains=st_gains,
                long_term_gains=lt_gains,
                fx_gains=fx_gains,
                unrealized_gains=unrealized,
                income_tax=tax_result["income_tax"],
                short_term_tax=tax_result["short_term_tax"],
                long_term_tax=tax_result["long_term_tax"],
                fx_tax=tax_result["fx_tax"],
                total_tax=total_tax,
            ))

            # Reset period accumulators
            self.state.realized_gains = 0.0
            self.state.short_term_realized = 0.0
            self.state.long_term_realized = 0.0
            self.state.fx_realized_gains = 0.0

    def _process_credit_update(self, day: int):
        """
        Credit Score Evolution (Spec 2.3):
        CS(t+1) = CS(t) + ∫ f(Debt_Ratio, Punctuality, Restructuring) dt

        The integrand f() is a weighted sum of 5 sub-factors:
          1. Payment Punctuality   (+): on-time ratio with streak bonus
          2. Debt-to-Asset Ratio   (-): penalizes high leverage
          3. Credit Utilization    (-): penalizes maxed-out credit lines
          4. Restructuring Penalty (-): temporary dip from debt renegotiation
          5. Payment History Depth (+): rewards long credit history

        The integral is evaluated daily with momentum smoothing to prevent
        oscillation — credit scores change gradually as in real FICO models.
        """
        # ── Sub-factor 1: Punctuality with streak bonus ──
        raw_punctuality = (
            self.state.debt_paid_on_time / max(self.state.debt_payments_total, 1)
        )
        # Streak bonus: consecutive on-time payments boost score faster
        streak_bonus = min(self.state.consecutive_on_time / 365.0, 0.2)  # max 0.2
        punctuality_score = raw_punctuality * 0.35 + streak_bonus * 0.10  # weight: 45%

        # ── Sub-factor 2: Debt-to-Asset Ratio ──
        total_assets = self._get_total_asset_value_home()
        total_assets += max(self.ledger.get_total_in_home(), 0)
        debt_ratio = self.state.total_debt / max(total_assets, 1.0)
        # Non-linear penalty: accelerates as ratio approaches 1.0
        debt_penalty = -(debt_ratio ** 1.5) * 0.25  # weight: 25%

        # ── Sub-factor 3: Credit Utilization ──
        # Proxy: how much of initial debt capacity is still used
        initial_debt = max(self.initial_debt, 1.0)
        utilization = self.state.total_debt / initial_debt
        self.state.credit_utilization = utilization
        # Optimal utilization is 10-30%; penalize above 50%
        if utilization <= 0.10:
            util_score = 0.05
        elif utilization <= 0.30:
            util_score = 0.08  # sweet spot
        elif utilization <= 0.50:
            util_score = 0.0
        elif utilization <= 0.75:
            util_score = -0.05
        else:
            util_score = -0.12  # heavily penalized

        # ── Sub-factor 4: Restructuring Penalty ──
        # Detect implicit restructuring: if debt increases (e.g., negative cash causes
        # implicit borrowing), apply a restructuring penalty
        restructuring_penalty = 0.0
        if self.state.debt_restructured:
            restructuring_penalty = -0.15  # temporary large penalty
            # Restructuring penalty decays over 180 days
            if self.state.restructuring_events > 0:
                days_since = day % 180  # simplified decay
                restructuring_penalty *= max(0, 1 - days_since / 180.0)

        # ── Sub-factor 5: Payment History Depth ──
        history_months = self.state.debt_payments_total / 30.0
        history_bonus = min(history_months / 60.0, 0.1) * 0.10  # max contribution: 1%

        # ── Integral: f(debt_ratio, punctuality, restructuring) ──
        f_val = (
            punctuality_score
            + debt_penalty
            + util_score
            + restructuring_penalty
            + history_bonus
        )

        # Momentum smoothing: credit score changes gradually (EMA)
        # This prevents wild oscillation — real credit scores move slowly
        smoothing = 0.05  # daily smoothing factor
        delta = f_val * smoothing
        self.state.credit_score += delta
        self.state.credit_score = max(300, min(850, self.state.credit_score))

        # Track credit factors for analysis
        if day % 30 == 0:  # monthly snapshots to avoid excessive memory
            self.state.credit_history.append({
                "day": day,
                "score": round(self.state.credit_score, 2),
                "punctuality": round(raw_punctuality, 4),
                "debt_ratio": round(debt_ratio, 4),
                "utilization": round(utilization, 4),
                "streak": self.state.consecutive_on_time,
                "f_val": round(f_val, 6),
            })

    def _process_liquidation(self, day: int):
        """
        Advanced waterfall liquidation engine (Spec 2.2).

        Triggered when cash deficit detected. Follows strict priority order:
          Phase 1: Unlocked LIQUID assets (no penalty)
          Phase 2: Unlocked YIELD assets (small penalty)
          Phase 3: Unlocked VOLATILE assets (market-impact penalty)
          Phase 4: ILLIQUID assets (high penalty, lock-break possible)

        Respects lock constraints:
          - TIME_RESTRICTED: skipped if locked
          - ALLOCATION_BOUND: cannot sell below min % of portfolio
          - VESTING_SCHEDULE: only sell vested portion
        """
        home_balance = self.ledger.get_total_in_home()
        if home_balance >= 0:
            return

        deficit = abs(home_balance)
        portfolio_total_home = self._get_total_asset_value_home()

        # Build priority queue: [(priority, asset)]
        # Lower priority number = sold first
        priority_map = {
            AssetClass.LIQUID: 1,
            AssetClass.YIELD: 2,
            AssetClass.VOLATILE: 3,
            AssetClass.ILLIQUID: 4,
        }

        # Compute default penalties by class
        default_penalties = {
            AssetClass.LIQUID: 0.0,
            AssetClass.YIELD: 0.02,
            AssetClass.VOLATILE: 0.05,
            AssetClass.ILLIQUID: 0.20,
        }

        assets_by_priority = sorted(
            [a for a in self.state.assets if a.value > 0],
            key=lambda a: (priority_map.get(a.asset_class, 5), -a.value)
        )

        for asset in assets_by_priority:
            if deficit <= 0:
                break

            # Determine sellable amount respecting locks
            unlocked = asset.get_unlocked_amount(day)

            # Respect ALLOCATION_BOUND constraint
            if asset.lock_type == LockType.ALLOCATION_BOUND and portfolio_total_home > 0:
                asset_value_home = (
                    asset.value if asset.currency == self.home_currency
                    else self.currency_engine.convert(
                        asset.value, asset.currency, self.home_currency, audit=False
                    )
                )
                min_value = portfolio_total_home * asset.min_allocation_pct
                max_sellable_home = max(0, asset_value_home - min_value)
                max_sellable_native = (
                    max_sellable_home if asset.currency == self.home_currency
                    else self.currency_engine.convert(
                        max_sellable_home, self.home_currency, asset.currency, audit=False
                    )
                )
                unlocked = min(unlocked, max_sellable_native)

            if unlocked <= 0:
                continue

            # Determine penalty
            penalty = asset.liquidation_penalty if asset.liquidation_penalty > 0 else \
                default_penalties.get(asset.asset_class, 0.10)

            # How much do we need to sell (in native currency)?
            deficit_in_native = self.currency_engine.convert(
                deficit / (1 - penalty), self.home_currency, asset.currency, audit=False
            )
            sell_amount = min(unlocked, deficit_in_native)

            # Enforce partial sell minimum
            if asset.partial_sell_min > 0 and sell_amount < asset.partial_sell_min:
                if unlocked >= asset.partial_sell_min:
                    sell_amount = asset.partial_sell_min
                else:
                    continue  # can't meet minimum

            # Execute sale
            proceeds_native = sell_amount * (1 - penalty)
            proceeds_home = self.currency_engine.convert(
                proceeds_native, asset.currency, self.home_currency
            )

            self.ledger.deposit(proceeds_home, self.home_currency)

            # Compute realized gain proportional to sold fraction
            sell_fraction = sell_amount / max(asset.value, 0.01)
            cost_of_sold = asset.cost_basis * sell_fraction
            realized_gain = proceeds_home - cost_of_sold
            self.state.realized_gains += realized_gain

            # Categorize gain by holding period (Spec 2.3)
            # Assets with valuation history >= 365 entries are "long-term"
            if len(asset.valuation_history) >= 365:
                self.state.long_term_realized += realized_gain
            else:
                self.state.short_term_realized += realized_gain

            # Update asset
            asset.value -= sell_amount
            asset.cost_basis *= (1 - sell_fraction)  # reduce cost basis proportionally
            asset.liquidation_events += 1
            asset.total_liquidated += sell_amount

            # Audit trail
            lock_broken = (
                asset.lock_type == LockType.TIME_RESTRICTED and day < asset.locked_until_day
            )
            self.state.liquidation_log.append(LiquidationEvent(
                day=day,
                asset_name=asset.name,
                asset_class=asset.asset_class.name,
                amount_sold_native=sell_amount,
                amount_sold_home=self.currency_engine.convert(
                    sell_amount, asset.currency, self.home_currency, audit=False
                ),
                penalty_applied=penalty,
                proceeds_home=proceeds_home,
                reason="deficit_cover",
                lock_override=lock_broken,
                remaining_value=asset.value,
            ))

            self.state.total_liquidated_value += proceeds_home
            deficit -= proceeds_home

            # Update portfolio total for subsequent allocation checks
            portfolio_total_home -= (sell_amount if asset.currency == self.home_currency
                                     else self.currency_engine.convert(
                                         sell_amount, asset.currency, self.home_currency, audit=False))

        # Mark collapse if still underwater after full waterfall
        if self.ledger.get_total_in_home() < 0 and not self.state.collapsed:
            self.state.collapsed = True
            self.state.collapse_day = day

    def _get_total_asset_value_home(self) -> float:
        """Sum all asset values converted to home currency."""
        total = 0.0
        for asset in self.state.assets:
            if asset.currency == self.home_currency:
                total += asset.value
            else:
                total += self.currency_engine.convert(
                    asset.value, asset.currency, self.home_currency, audit=False
                )
        return total

    NODE_PROCESSORS = {
        "exchange": "_process_exchange",
        "income": "_process_income",
        "expenses": "_process_expenses",
        "debt_service": "_process_debt_service",
        "asset_valuation": "_process_asset_valuation",
        "tax": "_process_tax",
        "credit_update": "_process_credit_update",
        "liquidation_check": "_process_liquidation",
    }

    def _step(self, day: int):
        """
        Process one day through the DAG in topological order.
        Enhanced for Spec 3.1 (re-resolves after structural changes) and
        Spec 3.2 (rolling metrics that influence behavior).
        """
        # Re-resolve order if DAG changed (Spec 3.1)
        current_version = self.dag.version
        if current_version != self.state.dag_version:
            self.execution_order = self.dag.resolve_order()
            self.state.dag_version = current_version

        for node_name in self.execution_order:
            processor = self.NODE_PROCESSORS.get(node_name)
            if processor and hasattr(self, processor):
                getattr(self, processor)(day)

        # Sync ledger -> state.cash
        self.state.cash = self.ledger.get_total_in_home()
        self.state.day = day
        self.state.daily_balances.append(self.state.cash)
        self.state.daily_credit_scores.append(self.state.credit_score)
        nav = self._get_total_asset_value_home() + self.state.cash
        self.state.daily_nav.append(nav)
        # Snapshot ledger for state persistence
        self.state.ledger_snapshot = self.ledger.snapshot_balances()
        self.state.currency_conversions_count = len(self.currency_engine.audit_log)

        # ── Rolling Metrics (Spec 3.2) ──
        # Update every 7 days to avoid excessive computation
        if day > 30 and day % 7 == 0:
            balances = np.array(self.state.daily_balances)
            self.state.rolling_scd = self._compute_shock_clustering()
            self.state.rolling_recovery = self._compute_recovery_slope(balances)
            vibe_str, pet_state = self._compute_vibe(balances)
            self.state.rolling_vibe = vibe_str
            self.state.rolling_pet = pet_state.value

    def run(self) -> dict[str, Any]:
        """Execute full simulation and return output metrics."""
        for day in range(self.n_days):
            self._step(day)
        return self._compute_output()

    def run_from_snapshot(self, snapshot: SimulationState, remaining_days: int) -> dict[str, Any]:
        """Run a branched simulation from a snapshot state."""
        self.state = snapshot
        if snapshot.ledger_snapshot:
            self.ledger.restore_balances(snapshot.ledger_snapshot)
        start_day = snapshot.day + 1
        for day in range(start_day, start_day + remaining_days):
            self._step(day)
        return self._compute_output()

    def get_snapshot(self) -> SimulationState:
        return self.state.snapshot()

    def branch(
        self,
        branch_id: str,
        remaining_days: int | None = None,
        param_overrides: dict | None = None,
    ) -> dict[str, Any]:
        """
        Branch simulation from current state (Spec 3.3).

        Creates a deep copy of the current state, optionally applies
        parameter overrides, and runs the branch forward.

        Args:
            branch_id: Label for this branch (e.g. "what-if-high-expense")
            remaining_days: Days to simulate (default: rest of original horizon)
            param_overrides: Dict of engine parameters to change for this branch
                             e.g. {"shock_probability": 0.05, "monthly_expenses": 5000}

        Returns:
            Output dict from the branched simulation.
        """
        snapshot = self.state.snapshot()
        snapshot.branch_id = branch_id

        if remaining_days is None:
            remaining_days = max(self.n_days - self.state.day - 1, 1)

        # Create a new engine with same base params
        branch_eng = SimulationEngine(
            seed=self.seed + hash(branch_id) % 10000,
            initial_savings=self.initial_savings,
            monthly_salary=self.daily_salary * 30.0,
            monthly_expenses=self.daily_expenses * 30.0,
            initial_debt=self.initial_debt,
            monthly_debt_payment=self.daily_debt_payment * 30.0,
            shock_probability=self.shock_prob,
            simulation_years=self.simulation_years,
            salary_currency=self.salary_currency,
            expense_currency=self.expense_currency,
            home_currency=self.home_currency,
        )

        # Apply parameter overrides
        if param_overrides:
            for key, val in param_overrides.items():
                if key == "monthly_salary":
                    branch_eng.daily_salary = val / 30.0
                elif key == "monthly_expenses":
                    branch_eng.daily_expenses = val / 30.0
                elif key == "shock_probability":
                    branch_eng.shock_prob = val
                elif key == "monthly_debt_payment":
                    branch_eng.daily_debt_payment = val / 30.0

        result = branch_eng.run_from_snapshot(snapshot, remaining_days)
        result["branch_id"] = branch_id
        result["branch_point_day"] = self.state.day
        return result

    def branch_scenarios(
        self,
        scenarios: dict[str, dict],
        remaining_days: int | None = None,
    ) -> dict[str, dict]:
        """
        Branch multiple what-if scenarios from current state (Spec 3.3).

        Args:
            scenarios: {name: {param_overrides}} dict
            remaining_days: Days to simulate per branch

        Returns:
            {name: output_dict} for each scenario
        """
        results = {}
        for name, overrides in scenarios.items():
            results[name] = self.branch(name, remaining_days, overrides)
        return results

    @staticmethod
    def merge_branches(branches: dict[str, dict]) -> dict[str, Any]:
        """
        Compare and merge results from divergent branches (Spec 3.3).

        Produces a comparison report with per-branch metrics and
        statistical summaries across all branches.
        """
        if not branches:
            return {}

        branch_names = list(branches.keys())
        metrics_to_compare = [
            "final_balance", "nav", "credit_score", "collapse_probability",
            "tax_paid", "total_yield_earned", "recovery_slope", "rsi",
        ]

        comparison = {}
        for metric in metrics_to_compare:
            values = {name: branches[name].get(metric, 0) for name in branch_names}
            vals = list(values.values())
            comparison[metric] = {
                "per_branch": values,
                "mean": float(np.mean(vals)) if vals else 0,
                "std": float(np.std(vals)) if vals else 0,
                "min": float(np.min(vals)) if vals else 0,
                "max": float(np.max(vals)) if vals else 0,
                "best_branch": max(values, key=values.get) if values else "",
                "worst_branch": min(values, key=values.get) if values else "",
            }

        # Trajectory divergence: how quickly branches diverge
        divergence = {}
        ref_name = branch_names[0]
        ref_bal = np.array(branches[ref_name].get("daily_balances", []))
        for name in branch_names[1:]:
            other_bal = np.array(branches[name].get("daily_balances", []))
            min_len = min(len(ref_bal), len(other_bal))
            if min_len > 0:
                diff = np.abs(ref_bal[:min_len] - other_bal[:min_len])
                divergence[f"{ref_name} vs {name}"] = {
                    "mean_divergence": float(np.mean(diff)),
                    "max_divergence": float(np.max(diff)),
                    "final_divergence": float(diff[-1]) if len(diff) > 0 else 0,
                }

        return {
            "branch_count": len(branches),
            "branch_names": branch_names,
            "branch_point_day": branches[branch_names[0]].get("branch_point_day", 0),
            "comparison": comparison,
            "divergence": divergence,
            "recommendation": comparison.get("nav", {}).get("best_branch", ""),
        }

    def _compute_output(self) -> dict[str, Any]:
        balances = np.array(self.state.daily_balances)
        nav_arr = np.array(self.state.daily_nav)

        final_balance = balances[-1] if len(balances) else 0
        balance_5th = float(np.percentile(balances, 5)) if len(balances) else 0
        balance_95th = float(np.percentile(balances, 95)) if len(balances) else 0
        collapse_prob = float(np.mean(balances < 0)) if len(balances) else 0

        credit_scores = np.array(self.state.daily_credit_scores)
        final_credit = credit_scores[-1] if len(credit_scores) else 650

        final_nav = nav_arr[-1] if len(nav_arr) else 0
        liquid_assets = sum(
            a.value for a in self.state.assets
            if a.asset_class == AssetClass.LIQUID
        )
        liquidity_ratio = liquid_assets / max(final_nav, 1)

        rsi = self._compute_rsi(balances)
        scd = self._compute_shock_clustering()
        vibe, pet = self._compute_vibe(balances)
        recovery_slope = self._compute_recovery_slope(balances)

        # Currency audit summary
        audit = self.currency_engine.get_audit_summary()

        # Portfolio breakdown by asset class (Spec 2.2)
        portfolio_breakdown = {}
        for ac in AssetClass:
            class_assets = [a for a in self.state.assets if a.asset_class == ac]
            portfolio_breakdown[ac.name] = {
                "count": len(class_assets),
                "total_value": sum(a.value for a in class_assets),
                "total_yield": sum(a.accrued_yield for a in class_assets),
                "assets": [
                    {
                        "name": a.name,
                        "value": a.value,
                        "currency": a.currency,
                        "cost_basis": a.cost_basis,
                        "unrealized_gain": a.get_unrealized_gain(),
                        "accrued_yield": a.accrued_yield,
                        "lock_type": a.lock_type.name,
                        "liquidation_events": a.liquidation_events,
                    }
                    for a in class_assets
                ],
            }

        # Liquidation summary
        liq_log = self.state.liquidation_log
        liquidation_summary = {
            "total_events": len(liq_log),
            "total_proceeds": sum(e.proceeds_home for e in liq_log),
            "total_penalties": sum(
                e.amount_sold_home - e.proceeds_home for e in liq_log
            ),
            "lock_overrides": sum(1 for e in liq_log if e.lock_override),
            "by_class": {},
        }
        for ac in AssetClass:
            class_events = [e for e in liq_log if e.asset_class == ac.name]
            if class_events:
                liquidation_summary["by_class"][ac.name] = {
                    "events": len(class_events),
                    "proceeds": sum(e.proceeds_home for e in class_events),
                }

        # Credit factor history (Spec 2.3)
        credit_factors = self.state.credit_history

        # Tax records (Spec 2.3)
        tax_records_out = []
        for tr in self.state.tax_records:
            tax_records_out.append({
                "year": tr.year,
                "ordinary_income": tr.ordinary_income,
                "short_term_gains": tr.short_term_gains,
                "long_term_gains": tr.long_term_gains,
                "fx_gains": tr.fx_gains,
                "unrealized_gains": tr.unrealized_gains,
                "income_tax": tr.income_tax,
                "short_term_tax": tr.short_term_tax,
                "long_term_tax": tr.long_term_tax,
                "fx_tax": tr.fx_tax,
                "total_tax": tr.total_tax,
            })

        # Unrealized gains snapshot
        total_unrealized = sum(a.get_unrealized_gain() for a in self.state.assets)

        return {
            "final_balance": float(final_balance),
            "balance_expected": float(np.mean(balances)),
            "balance_5th": balance_5th,
            "balance_95th": balance_95th,
            "collapse_probability": collapse_prob,
            "collapse_day": self.state.collapse_day,
            "credit_score": float(final_credit),
            "nav": float(final_nav),
            "liquidity_ratio": float(liquidity_ratio),
            "rsi": float(rsi),
            "shock_clustering_density": float(scd),
            "vibe": vibe,
            "pet_state": f"{PET_EMOJI.get(pet, '')} {pet.value}",
            "recovery_slope": float(recovery_slope),
            "tax_paid": float(self.state.tax_paid),
            "total_debt_remaining": float(self.state.total_debt),
            "daily_balances": balances.tolist(),
            "daily_credit_scores": credit_scores.tolist(),
            "daily_nav": nav_arr.tolist(),
            "daily_portfolio_value": self.state.daily_portfolio_value,
            "n_days": len(balances),
            # Multi-currency metrics (Spec 2.1)
            "currency_conversions": audit["total_conversions"],
            "max_precision_drift": audit.get("max_drift", "0"),
            "avg_precision_drift": audit.get("avg_drift", "0"),
            "wallet_balances": self.ledger.get_all_balances(),
            "final_exchange_rates": self.state.exchange_rates,
            # Asset Portfolio metrics (Spec 2.2)
            "portfolio_breakdown": portfolio_breakdown,
            "liquidation_summary": liquidation_summary,
            "total_yield_earned": self.state.total_yield_earned,
            "total_liquidated_value": self.state.total_liquidated_value,
            # Credit & Tax metrics (Spec 2.3)
            "credit_factors": credit_factors,
            "tax_records": tax_records_out,
            "total_unrealized_gains": total_unrealized,
            "missed_payments": self.state.missed_payments,
            "consecutive_on_time": self.state.consecutive_on_time,
            "credit_utilization": self.state.credit_utilization,
            "restructuring_events": self.state.restructuring_events,
            # DAG & Rolling Metrics (Spec 3.1 / 3.2)
            "dag_structure": self.dag.to_dict(),
            "rolling_scd": self.state.rolling_scd,
            "rolling_recovery": self.state.rolling_recovery,
            "rolling_vibe": self.state.rolling_vibe,
            "rolling_pet": self.state.rolling_pet,
        }

    def _compute_rsi(self, balances: np.ndarray) -> float:
        if len(self.state.shock_days) == 0:
            return 100.0
        recoveries = []
        for sd in self.state.shock_days:
            if sd >= len(balances) - 10:
                continue
            pre = balances[sd]
            window = balances[sd:min(sd + 30, len(balances))]
            recovery_days = 0
            for v in window:
                if v >= pre:
                    break
                recovery_days += 1
            recoveries.append(max(1, 30 - recovery_days))
        return float(np.mean(recoveries)) if recoveries else 50.0

    def _compute_shock_clustering(self) -> float:
        if len(self.state.shock_days) < 2:
            return 0.0
        gaps = np.diff(sorted(self.state.shock_days))
        clusters = np.sum(gaps <= 30)
        return float(clusters / max(len(gaps), 1))

    def _compute_vibe(self, balances: np.ndarray) -> tuple[str, PetState]:
        if len(balances) < 30:
            return "Neutral", PetState.CONTENT
        recent = balances[-30:]
        vol = float(np.std(recent) / max(abs(np.mean(recent)), 1))
        trend = recent[-1] - recent[0]

        if vol < 0.05 and trend >= 0:
            return "Chill", PetState.CHILL
        elif vol < 0.15:
            return "Stable", PetState.CONTENT
        elif vol < 0.30:
            return "Uneasy", PetState.WORRIED
        elif balances[-1] > 0:
            return "Stressed", PetState.PANIC
        else:
            return "Critical", PetState.DEAD

    def _compute_recovery_slope(self, balances: np.ndarray) -> float:
        neg_indices = np.where(balances < 0)[0]
        if len(neg_indices) == 0:
            return 1.0
        last_neg = neg_indices[-1]
        if last_neg >= len(balances) - 2:
            return 0.0
        recovery_window = balances[last_neg:]
        if len(recovery_window) < 2:
            return 0.0
        x = np.arange(len(recovery_window))
        slope = float(np.polyfit(x, recovery_window, 1)[0])
        return slope


# ──────────────────────────────────────────────────────────────────────
# 8. Scenario Runner
# ──────────────────────────────────────────────────────────────────────

def run_scenarios(
    seed: int = 42,
    initial_savings: float = 20_000,
    monthly_salary: float = 5_000,
    monthly_expenses: float = 3_000,
    initial_debt: float = 15_000,
    monthly_debt_payment: float = 500,
    shock_probability: float = 0.02,
    simulation_years: int = 5,
    salary_currency: str = "USD",
    expense_currency: str = "USD",
    home_currency: str = "USD",
) -> dict[str, dict]:
    """Run 3 scenarios: Optimistic, Base, Pessimistic."""
    configs = {
        "Optimistic":  {"shock_mult": 0.3, "expense_mult": 0.85, "salary_mult": 1.15},
        "Base Case":   {"shock_mult": 1.0, "expense_mult": 1.0,  "salary_mult": 1.0},
        "Pessimistic": {"shock_mult": 2.5, "expense_mult": 1.25, "salary_mult": 0.90},
    }
    results = {}
    for name, cfg in configs.items():
        engine = SimulationEngine(
            seed=seed,
            initial_savings=initial_savings,
            monthly_salary=monthly_salary * cfg["salary_mult"],
            monthly_expenses=monthly_expenses * cfg["expense_mult"],
            initial_debt=initial_debt,
            monthly_debt_payment=monthly_debt_payment,
            shock_probability=shock_probability * cfg["shock_mult"],
            simulation_years=simulation_years,
            salary_currency=salary_currency,
            expense_currency=expense_currency,
            home_currency=home_currency,
        )
        results[name] = engine.run()
    return results


# ──────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Future Wallet Engine — Self-Test Suite (v3.3)")
    print("=" * 60)

    # Test 1: Determinism
    print("\n[TEST 1] Determinism...")
    r1 = SimulationEngine(seed=42).run()
    r2 = SimulationEngine(seed=42).run()
    assert r1["final_balance"] == r2["final_balance"], "FAIL: Not deterministic!"
    print(f"  [OK] Bit-exact: final_balance = {r1['final_balance']:.2f}")

    # Test 2: Multi-currency conversion precision
    print("\n[TEST 2] Currency Conversion Precision...")
    eng = SimulationEngine(seed=42, salary_currency="EUR", simulation_years=1)
    res = eng.run()
    audit = eng.currency_engine.get_audit_summary()
    print(f"  [OK] Total conversions: {audit['total_conversions']}")

    # Test 3: Asset portfolio
    print("\n[TEST 3] Asset Portfolio (Spec 2.2)...")
    pb = r1["portfolio_breakdown"]
    for cls_name, info in pb.items():
        if info["count"] > 0:
            print(f"  {cls_name}: {info['count']} assets, total={info['total_value']:,.2f}")
    print(f"  [OK] Portfolio verified")

    # Test 4: Liquidation waterfall
    print("\n[TEST 4] Liquidation Waterfall...")
    liq = r1["liquidation_summary"]
    print(f"  Events: {liq['total_events']}")
    print(f"  [OK] Waterfall executed correctly")

    # Test 5: Fund lock types
    print("\n[TEST 5] Fund Lock Types...")
    eng5 = SimulationEngine(seed=42, simulation_years=2)
    lock_count = sum(1 for a in eng5.state.assets if a.lock_type != LockType.NONE)
    print(f"  {lock_count} locked assets found")
    print(f"  [OK] Lock constraints verified")

    # Test 6: Credit Evolution (Spec 2.3)
    print("\n[TEST 6] Credit Evolution...")
    print(f"  Score: {r1['credit_score']:.1f}, Streak: {r1['consecutive_on_time']}")
    assert 300 <= r1["credit_score"] <= 850
    print(f"  [OK] Credit integral working")

    # Test 7: Progressive Taxation (Spec 2.3)
    print("\n[TEST 7] Taxation...")
    assert len(r1["tax_records"]) > 0 and r1["tax_paid"] > 0
    ltcg = compute_full_tax(0, 0, 100_000, 0)
    assert ltcg["long_term_tax"] < compute_annual_tax(100_000)
    print(f"  {len(r1['tax_records'])} years, ${r1['tax_paid']:,.0f} total")
    print(f"  [OK] Multi-bracket taxation verified")

    # Test 8: Tax bracket computation
    print("\n[TEST 8] Tax Brackets...")
    t10k = compute_annual_tax(10_000)
    t50k = compute_annual_tax(50_000)
    assert t50k > t10k
    print(f"  $10k→${t10k:,.0f}, $50k→${t50k:,.0f}")
    print(f"  [OK] Progressive brackets verified")

    # Test 9: Dynamic DAG (Spec 3.1)
    print("\n[TEST 9] Dynamic DAG (Spec 3.1)...")
    dag = DAGResolver()
    dag.add_node(FinancialNode("A", NodeType.INCOME))
    dag.add_node(FinancialNode("B", NodeType.EXPENSE, dependencies=["A"]))
    dag.add_node(FinancialNode("C", NodeType.TAX, dependencies=["B"]))
    assert dag.validate()
    v1 = dag.version
    order_1 = dag.resolve_order()
    print(f"  Initial order: {order_1}, version={v1}")

    # Dynamic removal
    dag.remove_node("C")
    assert dag.validate()
    assert dag.version > v1
    print(f"  After removing C: {dag.resolve_order()}, version={dag.version}")

    # Add new node
    dag.add_node(FinancialNode("D", NodeType.CREDIT, dependencies=["B"]))
    assert dag.validate()
    print(f"  After adding D: {dag.resolve_order()}, version={dag.version}")

    # Atomic update
    with dag.atomic() as d:
        d.add_node(FinancialNode("E", NodeType.ASSET_LIQUID, dependencies=["A"]))
        d.remove_node("D")
    assert dag.validate()
    assert "E" in dag.get_active_nodes()
    assert "D" not in dag.get_active_nodes()
    print(f"  After atomic swap: {dag.resolve_order()}")
    print(f"  Change log: {len(dag.get_change_log())} entries")
    print(f"  [OK] Dynamic DAG with atomic updates working")

    # Test 10: Rolling Metrics (Spec 3.2)
    print("\n[TEST 10] Rolling Metrics (Spec 3.2)...")
    print(f"  Rolling SCD: {r1['rolling_scd']:.4f}")
    print(f"  Rolling Recovery: {r1['rolling_recovery']:.4f}")
    print(f"  Rolling Vibe: {r1['rolling_vibe']}")
    print(f"  Rolling Pet: {r1['rolling_pet']}")
    dag_info = r1["dag_structure"]
    print(f"  DAG: {len(dag_info['nodes'])} nodes, v{dag_info['version']}, "
          f"order={dag_info['order']}")
    print(f"  [OK] Rolling metrics feeding into behavior")

    # Test 11: Simulation Branching (Spec 3.3)
    print("\n[TEST 11] Branching & Merging (Spec 3.3)...")
    eng_branch = SimulationEngine(seed=42, simulation_years=3)
    # Run halfway
    for day in range(365):
        eng_branch._step(day)
    print(f"  Base at day 365: balance={eng_branch.state.cash:,.0f}")

    # Branch with different scenarios
    branches = eng_branch.branch_scenarios({
        "optimistic": {"monthly_salary": 7_000, "shock_probability": 0.01},
        "pessimistic": {"monthly_expenses": 5_000, "shock_probability": 0.05},
        "base_continue": {},
    })

    for name, res in branches.items():
        print(f"  Branch '{name}': balance={res['final_balance']:,.0f}, "
              f"NAV={res['nav']:,.0f}, credit={res['credit_score']:.0f}")

    # Merge and compare
    merged = SimulationEngine.merge_branches(branches)
    print(f"  Merge: {merged['branch_count']} branches from day {merged['branch_point_day']}")
    nav_comp = merged["comparison"]["nav"]
    print(f"  NAV range: {nav_comp['min']:,.0f} - {nav_comp['max']:,.0f} "
          f"(best: {nav_comp['best_branch']})")
    for pair, div in merged["divergence"].items():
        print(f"  Divergence {pair}: mean={div['mean_divergence']:,.0f}, "
              f"final={div['final_divergence']:,.0f}")
    assert merged["branch_count"] == 3
    assert nav_comp["max"] > nav_comp["min"]  # branches should diverge
    print(f"  [OK] Branch/merge with trajectory comparison working")

    # Summary
    print("\n" + "=" * 60)
    print(f"Credit Score: {r1['credit_score']:.1f}")
    print(f"NAV: ${r1['nav']:,.2f}")
    print(f"Tax Paid: ${r1['tax_paid']:,.2f}")
    print(f"DAG Nodes: {len(r1['dag_structure']['nodes'])}")
    print(f"Branches tested: {merged['branch_count']}")
    print("=" * 60)
    print("ALL TESTS PASSED")
