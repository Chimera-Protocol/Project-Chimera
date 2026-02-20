# src/csl_guardian.py
"""
CSL-Core backed Guardian — drop-in replacement for SymbolicGuardianV4/V6.

Same interface, same return types, same default parameters.
Validation rules live in .csl files, verified by Z3 at load time.
Computed fields and repair logic stay in Python.

Usage:
    guardian = CSLGuardianUnified()
    result = guardian.validate_action(action, current_state)
    # Returns: {"is_valid": bool, "message": str, optional "adjusted_amount": float}
"""

import os
from typing import Dict, Any, Tuple, Optional

# --- CSL-Core Import with Fallback ---
try:
    from chimera_core import load_guard, RuntimeConfig
    CSL_AVAILABLE = True
except ImportError:
    CSL_AVAILABLE = False
    import warnings
    warnings.warn(
        "csl-core not installed. Falling back to legacy SymbolicGuardian. "
        "Install with: pip install csl-core",
        ImportWarning,
        stacklevel=2,
    )

POLICY_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "policies")
COST_PER_ITEM = 50.0


# ============================================================
# E-Commerce Guardian (replaces SymbolicGuardianV4)
# ============================================================

class CSLGuardianEcom:
    """
    CSL-Core replacement for SymbolicGuardianV4.
    Identical __init__ signature and validate_action/repair_action interface.
    """

    def __init__(
        self,
        max_discount_per_week: float = 0.40,
        max_price_increase_per_week: float = 0.50,
        min_profit_margin_percentage: float = 0.15,
        max_price: float = 150.0,
        unit_cost: float = COST_PER_ITEM,
        ad_absolute_cap: float = 5000.0,
        ad_increase_cap: float = 1000.0,
        safety_buffer_ratio: float = 0.01,
        safety_buffer_abs: float = 0.0,
    ):
        self.cfg = dict(
            max_dn=max_discount_per_week,
            max_up=max_price_increase_per_week,
            min_margin=min_profit_margin_percentage,
            max_price=max_price,
            unit_cost=unit_cost,
            ad_cap=ad_absolute_cap,
            ad_increase_cap=ad_increase_cap,
            safety_buffer_ratio=safety_buffer_ratio,
            safety_buffer_abs=safety_buffer_abs,
        )

        # Load CSL policy — Z3 verifies at load time
        self._guard = None
        if CSL_AVAILABLE:
            policy_path = os.path.join(POLICY_DIR, "ecommerce_guard.csl")
            if os.path.exists(policy_path):
                self._guard = load_guard(policy_path, config=RuntimeConfig(raise_on_block=False))

    # --------------------------------------------------
    # Computed fields: float world → integer world for CSL
    # --------------------------------------------------
    def _prepare_csl_input(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert float action/state into CSL-compatible integer variables."""
        c = self.cfg
        price_change = float(action.get("price_change", 0.0) or 0.0)
        ad_spend = float(action.get("ad_spend", 0.0) or 0.0)
        prev_ad = float(current_state.get("weekly_ad_spend", 0.0) or 0.0)

        # Derived values
        current_price = float(current_state.get("price", 0.0))
        future_price = current_price * (1.0 + price_change)
        min_safe_price = c["unit_cost"] / (1.0 - c["min_margin"])
        min_safe_price_with_buffer = min_safe_price * (1.0 + c["safety_buffer_ratio"]) + c["safety_buffer_abs"]

        return {
            "ad_spend": int(round(ad_spend)),
            "prev_ad_spend": int(round(prev_ad)),
            "ad_increase": int(round(max(0, ad_spend - prev_ad))),
            "price_change_offset": int(round(price_change * 100)) + 100,
            "future_price_cents": int(round(future_price * 100)),
            "unit_cost_cents": int(round(c["unit_cost"] * 100)),
            "min_safe_price_cents": int(round(min_safe_price_with_buffer * 100)),
        }

    def _violation_to_message(self, violations: list, action: Dict, current_state: Dict) -> str:
        """Map CSL constraint names to human-readable messages matching legacy format."""
        c = self.cfg
        if not violations:
            return "Action is valid and compliant with all rules."

        # Use first violation for message (legacy behavior: return on first fail)
        name = violations[0] if isinstance(violations[0], str) else str(violations[0])

        messages = {
            "no_negative_ad": "Rule Violation: Ad spend cannot be negative.",
            "ad_absolute_cap": f"Rule Violation: Weekly ad spend cannot exceed ${c['ad_cap']}.",
            "ad_weekly_increase_cap": f"Rule Violation: Weekly ad spend increase cannot exceed ${c['ad_increase_cap']} vs last week.",
            "max_discount": f"Rule Violation: Weekly discount cannot exceed {c['max_dn']*100:.0f}%.",
            "max_price_increase": f"Rule Violation: Weekly price increase cannot exceed {c['max_up']*100:.0f}%.",
            "price_ceiling": f"Rule Violation: Price cannot exceed ${c['max_price']}.",
        }

        if name in ("above_cost", "min_margin_threshold"):
            price_change = float(action.get("price_change", 0.0) or 0.0)
            future_price = float(current_state.get("price", 0.0)) * (1.0 + price_change)
            min_safe = c["unit_cost"] / (1.0 - c["min_margin"])
            min_safe_buf = min_safe * (1.0 + c["safety_buffer_ratio"]) + c["safety_buffer_abs"]
            return (
                f"Rule Violation: Profit margin < {c['min_margin']*100:.0f}% "
                f"or price below safe threshold (${min_safe_buf:.2f} with buffer)."
            )

        return messages.get(name, f"Rule Violation: Constraint '{name}' violated.")

    # --------------------------------------------------
    # Public API — identical to SymbolicGuardianV4
    # --------------------------------------------------
    def validate_action(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        if self._guard is None:
            return self._validate_legacy(action, current_state)

        csl_input = self._prepare_csl_input(action, current_state)
        result = self._guard.verify(csl_input)

        if result.allowed:
            return {"is_valid": True, "message": "Action is valid and compliant with all rules."}
        else:
            violated = list(result.violations) if result.violations else []
            msg = self._violation_to_message(violated, action, current_state)
            return {"is_valid": False, "message": msg}

    def repair_action(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Clip action to nearest valid values, then validate via CSL."""
        c = self.cfg
        a = dict(
            price_change=float(action.get("price_change", 0.0) or 0.0),
            ad_spend=float(action.get("ad_spend", 0.0) or 0.0),
        )
        prev_ad = float(current_state.get("weekly_ad_spend", 0.0) or 0.0)

        # Ad spend: negative → 0; absolute cap; relative cap
        a["ad_spend"] = max(0.0, a["ad_spend"])
        a["ad_spend"] = min(c["ad_cap"], a["ad_spend"])
        if a["ad_spend"] - prev_ad > c["ad_increase_cap"]:
            a["ad_spend"] = prev_ad + c["ad_increase_cap"]

        # Price change percentage
        a["price_change"] = max(-c["max_dn"], min(c["max_up"], a["price_change"]))

        # Future price ceiling and minimum margin
        base_price = float(current_state["price"])
        future_price = base_price * (1.0 + a["price_change"])

        if future_price > c["max_price"]:
            a["price_change"] = (c["max_price"] / base_price) - 1.0
            future_price = c["max_price"]

        min_safe_price = c["unit_cost"] / (1.0 - c["min_margin"])
        min_safe_price_with_buffer = min_safe_price * (1.0 + c["safety_buffer_ratio"]) + c["safety_buffer_abs"]

        margin = (future_price - c["unit_cost"]) / max(1e-6, future_price)
        if future_price <= c["unit_cost"] or future_price < min_safe_price_with_buffer or margin < c["min_margin"]:
            a["price_change"] = (min_safe_price_with_buffer / base_price) - 1.0

        report = self.validate_action(a, current_state)
        return a, report

    # --------------------------------------------------
    # Legacy fallback (no csl-core installed)
    # --------------------------------------------------
    def _validate_legacy(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Exact copy of SymbolicGuardianV4.validate_action for fallback."""
        c = self.cfg
        price_change = float(action.get("price_change", 0.0) or 0.0)
        ad_spend = float(action.get("ad_spend", 0.0) or 0.0)
        prev_ad = float(current_state.get("weekly_ad_spend", 0.0) or 0.0)

        if ad_spend < 0:
            return {"is_valid": False, "message": "Rule Violation: Ad spend cannot be negative."}
        if ad_spend > c["ad_cap"]:
            return {"is_valid": False, "message": f"Rule Violation: Weekly ad spend cannot exceed ${c['ad_cap']}."}
        if ad_spend - prev_ad > c["ad_increase_cap"]:
            return {"is_valid": False, "message": f"Rule Violation: Weekly ad spend increase cannot exceed ${c['ad_increase_cap']} vs last week."}
        if price_change < 0 and abs(price_change) > c["max_dn"]:
            return {"is_valid": False, "message": f"Rule Violation: Weekly discount cannot exceed {c['max_dn']*100:.0f}%."}
        if price_change > 0 and price_change > c["max_up"]:
            return {"is_valid": False, "message": f"Rule Violation: Weekly price increase cannot exceed {c['max_up']*100:.0f}%."}

        future_price = float(current_state["price"]) * (1.0 + price_change)
        if future_price > c["max_price"]:
            return {"is_valid": False, "message": f"Rule Violation: Price cannot exceed ${c['max_price']}."}

        margin = (future_price - c["unit_cost"]) / max(1e-6, future_price)
        min_safe = c["unit_cost"] / (1.0 - c["min_margin"])
        min_safe_buf = min_safe * (1.0 + c["safety_buffer_ratio"]) + c["safety_buffer_abs"]

        if future_price <= c["unit_cost"] or future_price < min_safe_buf or margin < c["min_margin"]:
            return {
                "is_valid": False,
                "message": (
                    f"Rule Violation: Profit margin < {c['min_margin']*100:.0f}% "
                    f"or price below safe threshold (${min_safe_buf:.2f} with buffer)."
                ),
            }
        return {"is_valid": True, "message": "Action is valid and compliant with all rules."}


# ============================================================
# Unified Guardian (replaces SymbolicGuardianV6)
# ============================================================

class CSLGuardianUnified:
    """
    CSL-Core replacement for SymbolicGuardianV6.
    Routes e-commerce actions to CSLGuardianEcom,
    quant actions to CSL-backed quant validation.
    Same __init__ signature, same validate_action interface.
    """

    def __init__(
        self,
        # --- E-commerce ---
        max_discount_per_week: float = 0.40,
        max_price_increase_per_week: float = 0.50,
        min_profit_margin_percentage: float = 0.15,
        max_price: float = 150.0,
        unit_cost: float = COST_PER_ITEM,
        ad_absolute_cap: float = 5000.0,
        ad_increase_cap: float = 1000.0,
        # --- Quant ---
        allow_shorting: bool = True,
        max_position_ratio: float = 0.95,
        max_short_ratio: float = 0.50,
        max_action_amount: float = 1.0,
    ):
        # E-commerce sub-guardian
        self._ecom = CSLGuardianEcom(
            max_discount_per_week=max_discount_per_week,
            max_price_increase_per_week=max_price_increase_per_week,
            min_profit_margin_percentage=min_profit_margin_percentage,
            max_price=max_price,
            unit_cost=unit_cost,
            ad_absolute_cap=ad_absolute_cap,
            ad_increase_cap=ad_increase_cap,
        )

        # Quant config
        self.cfg = dict(
            allow_shorting=allow_shorting,
            max_pos_ratio=max_position_ratio,
            max_short_ratio=max_short_ratio,
            max_act_amount=max_action_amount,
        )

        # Load quant CSL policy
        self._quant_guard = None
        if CSL_AVAILABLE:
            policy_path = os.path.join(POLICY_DIR, "quant_guard.csl")
            if os.path.exists(policy_path):
                self._quant_guard = load_guard(policy_path, config=RuntimeConfig(raise_on_block=False))

    # --------------------------------------------------
    # Quant: computed helpers (stay in Python)
    # --------------------------------------------------
    def _max_buy_amount(self, cash: float, shares_held: float, price: float) -> float:
        if price <= 0 or cash < 0:
            return 0.0
        if shares_held < 0:
            return 1.0

        target = self.cfg["max_pos_ratio"]

        def ratio_for_amt(amt: float) -> float:
            cash_spend = cash * amt
            if price <= 0:
                return float("inf")
            sh_buy = cash_spend / price
            f_sh = shares_held + sh_buy
            f_cash = cash - cash_spend
            f_pv = f_cash + f_sh * price
            if f_pv <= 0:
                return float("inf")
            return (f_sh * price) / f_pv

        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = (lo + hi) / 2
            if ratio_for_amt(mid) <= target:
                lo = mid
            else:
                hi = mid

        return min(lo, self.cfg["max_act_amount"])

    def _max_short_amount(self, cash: float, shares_held: float, price: float, pv: float) -> float:
        target = self.cfg["max_short_ratio"]
        if not self.cfg["allow_shorting"] or price <= 0 or pv <= 0:
            return 0.0
        base_cap = target
        if cash > 0 and cash < pv:
            base_cap = max(0.0, min(target * (cash / pv), target))
        return max(0.0, min(base_cap, self.cfg["max_act_amount"]))

    # --------------------------------------------------
    # Quant: CSL validation
    # --------------------------------------------------
    def _prepare_quant_csl_input(self, action: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Compute derived fields and build CSL input dict. Returns None if state is invalid."""
        action_type = str(action.get("type", "HOLD")).upper()
        amount = float(action.get("amount", 0.0))

        cash = float(state.get("cash", 0.0))
        shares_held = float(state.get("shares_held", 0.0))
        current_price = state.get("market_data", {}).get("Close")

        price_valid = current_price is not None and current_price > 0
        portfolio_value = cash + (shares_held * (current_price or 0))
        portfolio_valid = portfolio_value > 0

        # Compute max amounts
        if action_type == "BUY" and price_valid and portfolio_valid:
            max_amt = self._max_buy_amount(cash, shares_held, current_price)
        elif action_type == "SHORT" and price_valid and portfolio_valid:
            max_amt = self._max_short_amount(cash, shares_held, current_price, portfolio_value)
        elif action_type == "SELL":
            max_amt = self.cfg["max_act_amount"]
        else:
            max_amt = self.cfg["max_act_amount"]

        return {
            "action_type": action_type,
            "amount_pct": int(round(amount * 100)),
            "max_amount_pct": int(round(max_amt * 100)),
            "has_long_position": "YES" if shares_held > 0 else "NO",
            "shorting_allowed": "YES" if self.cfg["allow_shorting"] else "NO",
            "portfolio_valid": "YES" if portfolio_valid else "NO",
            "price_valid": "YES" if price_valid else "NO",
            # Stash for auto-clip
            "_max_amt_float": max_amt,
        }

    def _validate_quant_rules(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        action_type = str(action.get("type", "HOLD")).upper()
        amount = float(action.get("amount", 0.0))

        # Global bounds check (before CSL)
        if not 0.0 <= amount <= self.cfg["max_act_amount"]:
            return {
                "is_valid": False,
                "message": f"Rule Violation: Action 'amount' ({amount:.2f}) must be between 0.00 and {self.cfg['max_act_amount']:.2f}.",
            }

        if action_type == "HOLD":
            return {"is_valid": True, "message": "Action is valid."}

        prepared = self._prepare_quant_csl_input(action, state)
        max_amt = prepared.pop("_max_amt_float")

        # CSL validation
        if self._quant_guard is not None:
            result = self._quant_guard.verify(prepared)

            if not result.allowed:
                violated = list(result.violations) if result.violations else []
                return self._quant_violation_to_message(violated, action_type, state, max_amt)

            # CSL passed — check if clip needed
            if amount > max_amt and action_type in ("BUY", "SHORT"):
                ratio_key = "max_pos_ratio" if action_type == "BUY" else "max_short_ratio"
                return {
                    "is_valid": True,
                    "message": f"{action_type} clipped to {max_amt:.2f} to respect max {'long' if action_type == 'BUY' else 'short'} ratio {self.cfg[ratio_key]:.0%}.",
                    "adjusted_amount": max_amt,
                }

            return {"is_valid": True, "message": "Action is valid."}

        # Fallback: legacy validation
        return self._validate_quant_legacy(action, state)

    def _quant_violation_to_message(self, violated: list, action_type: str, state: Dict, max_amt: float) -> Dict[str, Any]:
        """Map quant CSL violations to legacy message format."""
        if not violated:
            return {"is_valid": False, "message": "Rule Violation: Unknown constraint violated."}

        name = violated[0]

        if name == "valid_portfolio":
            return {"is_valid": False, "message": "State Error: Non-positive portfolio value."}
        if name == "valid_price":
            return {"is_valid": False, "message": "State Error: Invalid market price for validation."}
        if name == "sell_requires_long":
            return {"is_valid": False, "message": "Rule Violation: No long position to SELL."}
        if name == "short_must_be_allowed":
            return {"is_valid": False, "message": "Rule Violation: Short-selling is disabled."}
        if name == "amount_within_limit":
            if action_type == "BUY":
                if max_amt <= 0.0:
                    return {"is_valid": False, "message": f"Rule Violation: BUY not allowed given max long ratio {self.cfg['max_pos_ratio']:.0%}."}
                return {
                    "is_valid": True,
                    "message": f"BUY clipped to {max_amt:.2f} to respect max long ratio {self.cfg['max_pos_ratio']:.0%}.",
                    "adjusted_amount": max_amt,
                }
            if action_type == "SHORT":
                if max_amt <= 0.0:
                    return {"is_valid": False, "message": "Rule Violation: SHORT not allowed given current cash/equity."}
                return {
                    "is_valid": True,
                    "message": f"SHORT clipped to {max_amt:.2f} to respect max short ratio {self.cfg['max_short_ratio']:.0%}.",
                    "adjusted_amount": max_amt,
                }

        return {"is_valid": False, "message": f"Rule Violation: Constraint '{name}' violated."}

    def _validate_quant_legacy(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy fallback — exact copy of SymbolicGuardianV6._validate_quant_rules."""
        action_type = str(action.get("type", "HOLD")).upper()
        amount = float(action.get("amount", 0.0))

        if not 0.0 <= amount <= self.cfg["max_act_amount"]:
            return {"is_valid": False, "message": f"Rule Violation: Action 'amount' ({amount:.2f}) must be between 0.00 and {self.cfg['max_act_amount']:.2f}."}
        if action_type == "HOLD":
            return {"is_valid": True, "message": "Action is valid."}

        cash = float(state.get("cash", 0.0))
        shares_held = float(state.get("shares_held", 0.0))
        current_price = state.get("market_data", {}).get("Close")
        if current_price is None or current_price <= 0:
            return {"is_valid": False, "message": "State Error: Invalid market price for validation."}
        pv = cash + (shares_held * current_price)
        if pv <= 0:
            return {"is_valid": False, "message": "State Error: Non-positive portfolio value."}

        if action_type == "BUY":
            max_amt = self._max_buy_amount(cash, shares_held, current_price)
            if max_amt <= 0.0:
                return {"is_valid": False, "message": f"Rule Violation: BUY not allowed given max long ratio {self.cfg['max_pos_ratio']:.0%}."}
            if amount > max_amt:
                return {"is_valid": True, "message": f"BUY clipped to {max_amt:.2f} to respect max long ratio {self.cfg['max_pos_ratio']:.0%}.", "adjusted_amount": max_amt}
            return {"is_valid": True, "message": "Action is valid."}
        if action_type == "SELL":
            if shares_held > 0:
                return {"is_valid": True, "message": "Action is valid."}
            return {"is_valid": False, "message": "Rule Violation: No long position to SELL."}
        if action_type == "SHORT":
            if not self.cfg["allow_shorting"]:
                return {"is_valid": False, "message": "Rule Violation: Short-selling is disabled."}
            max_amt = self._max_short_amount(cash, shares_held, current_price, pv)
            if max_amt <= 0.0:
                return {"is_valid": False, "message": "Rule Violation: SHORT not allowed given current cash/equity."}
            if amount > max_amt:
                return {"is_valid": True, "message": f"SHORT clipped to {max_amt:.2f} to respect max short ratio {self.cfg['max_short_ratio']:.0%}.", "adjusted_amount": max_amt}
            return {"is_valid": True, "message": "Action is valid."}

        return {"is_valid": False, "message": f"Rule Violation: Unknown action type '{action_type}'."}

    # --------------------------------------------------
    # Domain switchboard — identical to V6
    # --------------------------------------------------
    def validate_action(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        if "price_change" in action or "ad_spend" in action:
            return self._ecom.validate_action(action, current_state)
        elif str(action.get("type", "")).upper() in ("BUY", "SELL", "HOLD", "SHORT"):
            return self._validate_quant_rules(action, current_state)
        return {"is_valid": False, "message": f"Rule Violation: Unrecognized action domain '{action.get('type')}'."}

    # Expose repair_action for e-commerce (delegates to ecom sub-guardian)
    def repair_action(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self._ecom.repair_action(action, current_state)
