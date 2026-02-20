# components.py

import os
import json
import pickle
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import shap
from tqdm import tqdm
import streamlit as st

from .environments import BaseEnvironment, BaseMultiAgentEnvironment

from .config import FEATURE_COLS_DEFAULT

import warnings

from .csl_guardian import CSLGuardianEcom, CSLGuardianUnified
SymbolicGuardianV4 = CSLGuardianEcom
SymbolicGuardianV6 = CSLGuardianUnified

# -------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

COST_PER_ITEM = 50.0


# ----------------------------
# Data types
# ----------------------------

@dataclass
class Action:
    price_change: float = 0.0
    ad_spend: float = 0.0


# --- Legacy (kept for reference, no longer active) ---
class _LegacySymbolicGuardianV4:

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
        safety_buffer_abs: float = 0.0            
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
            safety_buffer_abs=safety_buffer_abs
        )
    # ... rest of the class is unchanged
    def validate_action(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        c = self.cfg
        price_change = float(action.get("price_change", 0.0) or 0.0)
        ad_spend = float(action.get("ad_spend", 0.0) or 0.0)
        prev_ad = float(current_state.get("weekly_ad_spend", 0.0) or 0.0)

        # Negative ad spend is prohibited
        if ad_spend < 0:
            return {"is_valid": False, "message": "Rule Violation: Ad spend cannot be negative."}

        # Absolute cap
        if ad_spend > c["ad_cap"]:
            return {
                "is_valid": False,
                "message": f"Rule Violation: Weekly ad spend cannot exceed ${c['ad_cap']}.",
            }

        # Relative increase cap
        if ad_spend - prev_ad > c["ad_increase_cap"]:
            return {
                "is_valid": False,
                "message": f"Rule Violation: Weekly ad spend increase cannot exceed ${c['ad_increase_cap']} vs last week.",
            }

        # Symmetric price percentage
        if price_change < 0 and abs(price_change) > c["max_dn"]:
            return {"is_valid": False, "message": f"Rule Violation: Weekly discount cannot exceed {c['max_dn']*100:.0f}%."}
        if price_change > 0 and price_change > c["max_up"]:
            return {"is_valid": False, "message": f"Rule Violation: Weekly price increase cannot exceed {c['max_up']*100:.0f}%."}

        # Future price and margin
        future_price = float(current_state["price"]) * (1.0 + price_change)
        if future_price > c["max_price"]:
            return {"is_valid": False, "message": f"Rule Violation: Price cannot exceed ${c['max_price']}."}

        margin = (future_price - c["unit_cost"]) / max(1e-6, future_price)
        min_safe_price = c["unit_cost"] / (1.0 - c["min_margin"])
        # NEW in V4: Apply safety buffer to validation threshold
        min_safe_price_with_buffer = min_safe_price * (1.0 + c["safety_buffer_ratio"]) + c["safety_buffer_abs"]

        if future_price <= c["unit_cost"] or future_price < min_safe_price_with_buffer or margin < c["min_margin"]:
            return {
                "is_valid": False,
                "message": (
                    f"Rule Violation: Profit margin < {c['min_margin']*100:.0f}% "
                    f"or price below safe threshold (${min_safe_price_with_buffer:.2f} with buffer)."
                ),
            }

        return {"is_valid": True, "message": "Action is valid and compliant with all rules."}

    def repair_action(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

        # Future price ceiling and minimum margin requirement
        base_price = float(current_state["price"])
        future_price = base_price * (1.0 + a["price_change"])

        # Max price 
        if future_price > c["max_price"]:
            a["price_change"] = (c["max_price"] / base_price) - 1.0
            future_price = c["max_price"]

        # Min margin with safety buffer
        margin = (future_price - c["unit_cost"]) / max(1e-6, future_price)
        min_safe_price = c["unit_cost"] / (1.0 - c["min_margin"])
        min_safe_price_with_buffer = min_safe_price * (1.0 + c["safety_buffer_ratio"]) + c["safety_buffer_abs"]

        if future_price <= c["unit_cost"] or future_price < min_safe_price_with_buffer or margin < c["min_margin"]:
            target_price = min_safe_price_with_buffer
            a["price_change"] = (target_price / base_price) - 1.0

        report = self.validate_action(a, current_state)
        return a, report


DEFAULT_TRUST_VALUE_MULTIPLIER = 120_000 


# ----------------------------
# EcommerceSimulatorV5
# ----------------------------

class EcommerceSimulatorV5(BaseEnvironment): # MODIFIED: Inherits from BaseEnvironment
    """
    A simulator with more realistic and stable economic principles.
        - Weekly advertising budget spending (not cumulative).
        - Numpy Generator-based seed management for repeatability.
        - Brand trust erosion and price/name effects.
        - Simple seasonality (optional).
    """

    def __init__(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        base_demand: float = 800.0,
        price_elasticity: float = 1.2,
        ad_log_scale: float = 0.3,
        trust_ad_gain: float = 0.01,
        trust_decay: float = 0.002,
        price_penalty_threshold: float = 150.0,
        price_upper: float = 150.0,
        price_lower: float = COST_PER_ITEM,
        ad_min: float = 0.0,
        ad_max: float = 5_000.0,
        seasonality_amp: float = 0.2,
        noise_sigma: float = 0.05,
    ):
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.params = dict(
            base_demand=base_demand,
            price_elasticity=price_elasticity,
            ad_log_scale=ad_log_scale,
            trust_ad_gain=trust_ad_gain,
            trust_decay=trust_decay,
            price_penalty_threshold=price_penalty_threshold,
            price_upper=price_upper,
            price_lower=price_lower,
            ad_min=ad_min,
            ad_max=ad_max,
            seasonality_amp=seasonality_amp,
            noise_sigma=noise_sigma,
        )

        if initial_state:
            self.state = initial_state.copy()
        else:
            self.state = {
                "week": 1,
                "price": 100.0,
                "weekly_ad_spend": 500.0,
                "brand_trust": 0.7,
                "sales_volume": 0,
                "profit": 0.0,
                "season_phase": 0,
            }
        # Initial calculation
        self._update_sales_and_profit()

    def reset(self, seed: Optional[int] = None, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resets the environment to its initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if initial_state:
            self.state = initial_state.copy()
        else:
            self.state = {
                "week": 1,
                "price": 100.0,
                "weekly_ad_spend": 500.0,
                "brand_trust": 0.7,
                "sales_volume": 0,
                "profit": 0.0,
                "season_phase": 0,
            }
        self._update_sales_and_profit()
        return self.state.copy()

    def get_state(self) -> Dict[str, Any]:
        """Returns a copy of the current environment state."""
        return self.state.copy()

    def get_state_string(self) -> str:
        return json.dumps(self.state, indent=2)

    def _seasonality_factor(self) -> float:
        amp = self.params["seasonality_amp"]
        if amp <= 0.0:
            return 1.0
        phase = self.state.get("season_phase", 0) % 52
        s = 1.0 + amp * np.sin(2 * np.pi * phase / 52.0)
        return max(0.75, s)

    def _update_sales_and_profit(self):
        p = self.params
        weekly_ad = np.clip(self.state["weekly_ad_spend"], p["ad_min"], p["ad_max"])
        ad_multiplier = 1.0 + np.log1p(weekly_ad / 1000.0) * p["ad_log_scale"]

        price = float(np.clip(self.state["price"], p["price_lower"], p["price_upper"]))
        price_factor = (max(1e-6, price) / 100.0) ** (-p["price_elasticity"])
        
        season_factor = self._seasonality_factor()
        noise = self.rng.normal(1.0, p["noise_sigma"])

        raw_demand = p["base_demand"] * self.state["brand_trust"] * ad_multiplier * price_factor * season_factor
        demand = max(0.0, raw_demand * max(0.8, min(1.2, noise)))
        self.state["sales_volume"] = int(demand)

        revenue = self.state["sales_volume"] * price
        cogs = self.state["sales_volume"] * COST_PER_ITEM
        op_cost = weekly_ad
        profit = revenue - (cogs + op_cost)
        self.state["profit"] = float(profit)

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]: # MODIFIED: Renamed from take_action
        """Applies an action and advances the simulation by one week."""
        p = self.params
        price_change = float(action.get("price_change", 0.0) or 0.0)
        ad_spend = float(action.get("ad_spend", 0.0) or 0.0)
        ad_spend = float(np.clip(ad_spend, p["ad_min"], p["ad_max"]))

        if price_change > 0.10:
            self.state["brand_trust"] *= (1.0 - 0.02)  
        elif price_change < -0.05:
            self.state["brand_trust"] *= (1.0 + 0.03)  

        new_price = self.state["price"] * (1.0 + price_change)
        new_price = float(np.clip(new_price, p["price_lower"], p["price_upper"]))
        self.state["price"] = new_price
        self.state["weekly_ad_spend"] = ad_spend

        if ad_spend > 0:
            self.state["brand_trust"] *= (1.0 + np.log1p(ad_spend / 1000.0) * p["trust_ad_gain"])

        self.state["brand_trust"] *= (1.0 - p["trust_decay"])
        self.state["brand_trust"] = float(np.clip(self.state["brand_trust"], 0.2, 1.0))

        self.state["week"] += 1
        self.state["season_phase"] = (self.state.get("season_phase", 0) + 1) % 52

        self._update_sales_and_profit()
        return self.state.copy()


# ----------------------------
# EcommerceSimulatorV7
# ----------------------------

class EcommerceSimulatorV7(BaseMultiAgentEnvironment): # MODIFIED: Inherits from BaseMultiAgentEnvironment
    """
    A multi-agent competitive market simulator.
    """
    def __init__(
        self,
        num_agents: int = 3,
        seed: Optional[int] = None,
        base_demand: float = 1000.0,
        price_elasticity: float = 1.2,
        ad_attraction_factor: float = 0.5,
        ad_log_scale: float = 0.3,
        trust_ad_gain: float = 0.015,
        trust_decay: float = 0.002,
        price_upper: float = 150.0,
        price_lower: float = COST_PER_ITEM,
        ad_min: float = 0.0,
        ad_max: float = 5_000.0,
        seasonality_amp: float = 0.2,
        noise_sigma: float = 0.05,
    ):
        self.num_agents = num_agents
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.params = {
            'base_demand': base_demand, 'price_elasticity': price_elasticity,
            'ad_attraction_factor': ad_attraction_factor,
            'ad_log_scale': ad_log_scale, 'trust_ad_gain': trust_ad_gain,
            'trust_decay': trust_decay, 'price_upper': price_upper,
            'price_lower': price_lower, 'ad_min': ad_min, 'ad_max': ad_max,
            'seasonality_amp': seasonality_amp, 'noise_sigma': noise_sigma,
        }
        self.week = 1
        self.season_phase = 0
        self.agent_states = self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Resets the simulation for all agents to their initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.week = 1
        self.season_phase = 0
        self.agent_states = []
        
        default_initial_state = {
            "price": 100.0,
            "weekly_ad_spend": 500.0,
            "brand_trust": 0.7,
            "sales_volume": 0,
            "profit": 0.0,
            "market_share": 1.0 / self.num_agents,
        }

        for i in range(self.num_agents):
            state = default_initial_state.copy()
            state['agent_id'] = i
            self.agent_states.append(state)
            
        return self.agent_states.copy()

    def get_state(self) -> Dict[str, Any]: # MODIFIED: Renamed from get_full_state
        """Returns the complete state of the simulation."""
        return {
            "week": self.week,
            "season_phase": self.season_phase,
            "agents": self.agent_states.copy()
        }

    def _apply_actions(self, actions: List[Dict[str, Any]]):
        """Applies individual agent actions to update their internal states."""
        p = self.params
        for i, action in enumerate(actions):
            state = self.agent_states[i]
            
            price_change = float(action.get("price_change", 0.0) or 0.0)
            new_price = state["price"] * (1.0 + price_change)
            state["price"] = float(np.clip(new_price, p['price_lower'], p['price_upper']))

            state["weekly_ad_spend"] = float(np.clip(action.get("ad_spend", 0.0), p['ad_min'], p['ad_max']))

            if price_change > 0.10:
                state["brand_trust"] *= (1.0 - 0.02)
            elif price_change < -0.05:
                state["brand_trust"] *= (1.0 + 0.03)
            
            if state["weekly_ad_spend"] > 0:
                state["brand_trust"] *= (1.0 + np.log1p(state["weekly_ad_spend"] / 1000.0) * p['trust_ad_gain'])
            
            state["brand_trust"] *= (1.0 - p['trust_decay'])
            state["brand_trust"] = float(np.clip(state["brand_trust"], 0.2, 1.0))

    def _update_market_and_sales(self):
        """Calculates total demand and distributes it based on market share."""
        p = self.params
        
        total_market_ad_spend = sum(s['weekly_ad_spend'] for s in self.agent_states)
        avg_market_price = sum(s['price'] for s in self.agent_states) / self.num_agents
        
        season_factor = 1.0 + p['seasonality_amp'] * np.sin(2 * np.pi * self.season_phase / 52.0)
        ad_multiplier = 1.0 + np.log1p(total_market_ad_spend / 1000.0) * p['ad_log_scale']
        price_factor = (avg_market_price / 100.0) ** (-p['price_elasticity'])
        noise = self.rng.normal(1.0, p['noise_sigma'])
        
        total_demand = int(p['base_demand'] * ad_multiplier * price_factor * season_factor * noise)
        
        attractiveness_scores = []
        for state in self.agent_states:
            price_attraction = state['price'] ** -p['price_elasticity']
            ad_attraction = 1 + np.log1p(state['weekly_ad_spend'] / 1000.0) * p['ad_attraction_factor']
            score = state['brand_trust'] * price_attraction * ad_attraction
            attractiveness_scores.append(score)
            
        total_attractiveness = sum(attractiveness_scores)
        if total_attractiveness == 0: total_attractiveness = 1.0

        for i, state in enumerate(self.agent_states):
            state['market_share'] = attractiveness_scores[i] / total_attractiveness
            state['sales_volume'] = int(total_demand * state['market_share'])
            revenue = state['sales_volume'] * state['price']
            cogs = state['sales_volume'] * COST_PER_ITEM
            op_cost = state['weekly_ad_spend']
            state['profit'] = revenue - (cogs + op_cost)

    def step(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]: # MODIFIED: Renamed from take_action
        """Takes a list of actions and resolves the turn for the entire market."""
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, but received {len(actions)}.")
            
        self._apply_actions(actions)
        self._update_market_and_sales()
        
        self.week += 1
        self.season_phase = (self.season_phase + 1) % 52
        
        return self.agent_states.copy()
    


# ----------------------------
# CausalEngineV6
# ----------------------------
class CausalEngineV6:

    def __init__(
        self,
        data_path: str = "initial_causal_data.pkl",
        force_regenerate: bool = False,
        generator_seed: int = 123,
        trust_multiplier: float = DEFAULT_TRUST_VALUE_MULTIPLIER,
        num_simulations: int = 500,
    ):
        self.data_path = data_path
        self.model = None
        self.trust_multiplier = trust_multiplier
        self.initial_train_history: Optional[pd.DataFrame] = None
        self.rng = np.random.default_rng(generator_seed)

        if os.path.exists(self.data_path) and not force_regenerate:
            print("-> Causal Engine V6: Loading existing startup data...")
            self._load_data()
        else:
            print("-> Causal Engine V6: Generating new initial data (may take a few minutes)...")
            guardian = SymbolicGuardianV4()
            sim = EcommerceSimulatorV5(seed=42) 
            self.initial_train_history = self._generate_initial_data(sim, guardian, num_simulations)
            self._save_data()

        print("   - Data is ready. Causal Forest model is being trained...")
        self._fit_model(self.initial_train_history)
        print("   - Causal Engine V6 initialization and training completed.")

    def _generate_initial_data(
        self,
        simulator: EcommerceSimulatorV5,
        guardian: SymbolicGuardianV4,
        num_simulations: int = 500,
        num_steps: int = 50,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        print("-> Causal Engine V6: Generating new, truly randomized initial data...")
        
        abs_max_discount = guardian.cfg['max_dn']
        abs_max_increase = guardian.cfg['max_up']
        abs_ad_cap = guardian.cfg['ad_cap']

        for i in tqdm(range(num_simulations), desc="Generating Initial Training Data"):
            simulator.reset(seed=i)
            for _ in range(num_steps):
                state_before = simulator.state.copy()

                price_change = float(self.rng.uniform(-abs_max_discount, abs_max_increase))
                ad_spend = float(self.rng.uniform(0, abs_ad_cap))

                action_for_sim = {
                    "price_change": price_change,
                    "ad_spend": ad_spend
                }
                
                # NOTE: This internal call uses the original name, which is fine as it's a temporary instance.
                state_after = simulator.step(action_for_sim)
                profit_change = float(state_after["profit"] - state_before["profit"])
                trust_change = float(state_after["brand_trust"] - state_before["brand_trust"])

                rows.append({
                    "initial_price": float(state_before["price"]),
                    "initial_brand_trust": float(state_before["brand_trust"]),
                    "initial_ad_spend": float(state_before["weekly_ad_spend"]),
                    "season_phase": int(state_before["season_phase"]),
                    "price_change": float(action_for_sim["price_change"]),
                    "ad_spend": float(action_for_sim["ad_spend"]),
                    "trust_adjusted_profit_change": profit_change + (trust_change * self.trust_multiplier),
                    "profit_change": profit_change,
                    "trust_change": trust_change,
                    "sales_change": float(state_after["sales_volume"] - state_before["sales_volume"]),
                })
        return pd.DataFrame(rows)

    def _save_data(self):
        with open(self.data_path, "wb") as f:
            pickle.dump(self.initial_train_history, f)
        print(f"   - Initial data was saved to file '{self.data_path}'.")

    def _load_data(self):
        with open(self.data_path, "rb") as f:
            self.initial_train_history = pickle.load(f)
        print(f"   - Initial data was loaded from file '{self.data_path}'.")

    def _fit_model(self, training_data: pd.DataFrame):
        Y = training_data["trust_adjusted_profit_change"].values
        T = training_data[["price_change", "ad_spend"]].values
        XW = training_data[["initial_price", "initial_brand_trust", "initial_ad_spend", "season_phase"]].values

        self.model = CausalForestDML(
            model_y=GradientBoostingRegressor(random_state=123),
            model_t=MultiOutputRegressor(GradientBoostingRegressor(random_state=123)),
            discrete_treatment=False,
            random_state=123,
        )
        self.model.fit(Y, T, X=XW, W=XW)

    def retrain(self, new_experience_history: List[Dict[str, Any]]):
        new_data = pd.DataFrame(new_experience_history)

        if not new_data.empty and "profit_change" in new_data.columns and "trust_change" in new_data.columns:
            new_data["trust_adjusted_profit_change"] = new_data["profit_change"] + (new_data["trust_change"] * self.trust_multiplier)

        combined_history = pd.concat([self.initial_train_history, new_data], ignore_index=True)
        print(f"-> Causal Engine is being retrained. Total data points: {len(combined_history)}")
        self._fit_model(combined_history)
        self.initial_train_history = combined_history
        print("   - Retraining completed.")

    def estimate_causal_effect(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Causal model is not trained.")

        X_context = pd.DataFrame([{
            "initial_price": float(context["price"]),
            "initial_brand_trust": float(context["brand_trust"]),
            "initial_ad_spend": float(context["weekly_ad_spend"]), 
            "season_phase": int(context.get("season_phase", 0)),
        }])

        T0 = np.array([[0.0, 0.0]])
        T1 = np.array([[float(action.get("price_change", 0.0) or 0.0),
                        float(action.get("ad_spend", 0.0) or 0.0)]])

        effect = self.model.effect(X_context, T0=T0, T1=T1)
        
        return {"estimated_long_term_value": float(effect[0])}

    def simulate_plan(
        self,
        simulator: EcommerceSimulatorV5,
        guardian: SymbolicGuardianV4,
        start_state: Optional[Dict[str, Any]],
        plan: List[Dict[str, Any]],
        n_rollouts: int = 64,
        seed: Optional[int] = 999,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        profits = []
        sales_list = []
        trust_list = []

        for r in range(n_rollouts):
            if start_state is not None:
                s0 = start_state.copy()
                sim = EcommerceSimulatorV5(initial_state=s0, seed=int(rng.integers(0, 1_000_000)))
            else:
                sim = EcommerceSimulatorV5(seed=int(rng.integers(0, 1_000_000)))

            profit_acc = 0.0
            sales_acc = 0.0
            trust_end = sim.state["brand_trust"]

            for step_action in plan:
                safe_action, _ = guardian.repair_action(step_action, sim.state)
                before = sim.state.copy()
                # NOTE: This internal call uses the original name, which is fine as it's a temporary instance.
                after = sim.step(safe_action)
                profit_acc += float(after["profit"] - before["profit"])
                sales_acc += float(after["sales_volume"])
                trust_end = float(after["brand_trust"])

            profits.append(profit_acc)
            sales_list.append(sales_acc)
            trust_list.append(trust_end)

        def _summary(arr: List[float]) -> Dict[str, float]:
            a = np.asarray(arr, dtype=float)
            return {
                "mean": float(np.mean(a)),
                "p05": float(np.percentile(a, 5)),
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
            }

        return {
            "profit": _summary(profits),
            "sales": _summary(sales_list),
            "final_trust": _summary(trust_list),
            "horizon": len(plan),
            "rollouts": n_rollouts,
        }
    

    def explain_decision(self, context: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Causal model is not trained.")

        feature_names = ["initial_price", "initial_brand_trust", "initial_ad_spend", "season_phase", "price_change", "ad_spend"]

        instance_data = {
            "initial_price": context.get("price", 0.0),
            "initial_brand_trust": context.get("brand_trust", 0.0),
            "initial_ad_spend": context.get("weekly_ad_spend", 0.0),
            "season_phase": int(context.get("season_phase", 0)),
            "price_change": action.get("price_change", 0.0),
            "ad_spend": action.get("ad_spend", 0.0)
        }
        instance_df = pd.DataFrame([instance_data], columns=feature_names)
        instance_numpy = instance_df.values

        def prediction_wrapper(data_numpy):
            X_context = data_numpy[:, :4]
            T1_action = data_numpy[:, 4:]
            T0_noop = np.zeros_like(T1_action)
            return self.model.effect(X_context, T0=T0_noop, T1=T1_action)

        background_data_sample = self.initial_train_history.sample(min(100, len(self.initial_train_history)))
        
        background_data_numpy = background_data_sample[feature_names].values

        explainer = shap.Explainer(prediction_wrapper, background_data_numpy)
        
        shap_values_obj = explainer(instance_numpy)

        explanation = {
            "features": feature_names,
            "shap_values": shap_values_obj.values[0].tolist(),
            "base_value": shap_values_obj.base_values[0]
        }
        
        return explanation
    
    
# ----------------------------
# CausalEngineV6UseReady
# ----------------------------
class CausalEngineV6_UseReady:
    """
    Loads a pre-trained model file for fast, production-ready performance.
    Also handles retraining and SHAP-based explanations.
    """
    def __init__(
        self,
        trust_multiplier: float = DEFAULT_TRUST_VALUE_MULTIPLIER,
    ):
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.model_path = os.path.join(PROJECT_ROOT, 'models', 'trained_causal_model.pkl')
        
        self.model = None
        self.trust_multiplier = trust_multiplier
        self.background_data = self._load_background_data()

        if not os.path.exists(self.model_path):
            error_message = (f"Model file ('{self.model_path}') not found! Please run 'train_and_save_model.py' first.")
            raise FileNotFoundError(error_message)
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("CausalEngine: Model loaded successfully.")

    def _load_background_data(self, num_samples=100):
        data = {'initial_price': np.random.uniform(80, 150, num_samples), 'initial_brand_trust': np.random.uniform(0.2, 0.9, num_samples),
                'initial_ad_spend': np.random.uniform(0, 5000, num_samples), 'season_phase': np.random.randint(0, 52, num_samples),
                'price_change': np.random.uniform(-0.4, 0.5, num_samples), 'ad_spend': np.random.uniform(0, 5000, num_samples)}
        return pd.DataFrame(data)

    def retrain(self, new_experience_history: List[Dict[str, Any]]):
        st.toast("Note: Online learning is not yet implemented.", icon="ℹ️")
        pass
    
    def _simulate_trust_change(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        trust_change = 0.0
        ad_spend = action.get("ad_spend", 0.0)
        price_change = action.get("price_change", 0.0)
        if ad_spend > 0: trust_change += np.log1p(ad_spend / 1000.0) * 0.01
        if price_change > 0.10: trust_change -= 0.02
        elif price_change < -0.05: trust_change += 0.03
        return trust_change

    def estimate_causal_effect(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Causal model is not loaded.")

        X_context = pd.DataFrame([{
            "initial_price": float(context["price"]),
            "initial_brand_trust": float(context["brand_trust"]),
            "initial_ad_spend": float(context["weekly_ad_spend"]),
            "season_phase": int(context.get("season_phase", 0))
            }])

        T_action = np.array([[float(action.get("price_change", 0.0)), float(action.get("ad_spend", 0.0))]])
        T0_noop = np.zeros_like(T_action)

        estimated_profit_effect = self.model.effect(X_context, T0=T0_noop, T1=T_action)
        simulated_trust_change = self._simulate_trust_change(action, context)

        long_term_value = float(estimated_profit_effect[0]) + (simulated_trust_change * self.trust_multiplier)

        return {
            "estimated_long_term_value": long_term_value,
            "predicted_short_term_profit": float(estimated_profit_effect[0]),  
            "estimated_profit_change": float(estimated_profit_effect[0]),
            "simulated_trust_change": simulated_trust_change
            }


    def explain_decision(self, context: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None: raise RuntimeError("Causal model is not loaded.")
        feature_names = ["initial_price", "initial_brand_trust", "initial_ad_spend", "season_phase", "price_change", "ad_spend"]
        instance_data = {"initial_price": context.get("price", 0.0), "initial_brand_trust": context.get("brand_trust", 0.0),
                         "initial_ad_spend": context.get("weekly_ad_spend", 0.0), "season_phase": int(context.get("season_phase", 0)),
                         "price_change": action.get("price_change", 0.0), "ad_spend": action.get("ad_spend", 0.0)}
        
        instance_df = pd.DataFrame([instance_data], columns=feature_names)

        def prediction_wrapper(data): 
            X = data[["initial_price", "initial_brand_trust", "initial_ad_spend", "season_phase"]]
            T1_action = data[["price_change", "ad_spend"]]
            T0_noop = np.zeros_like(T1_action.values)
            return self.model.effect(X, T0=T0_noop, T1=T1_action.values)

        background_data_df = self.background_data[feature_names]
        explainer = shap.Explainer(prediction_wrapper, background_data_df)
        shap_values_obj = explainer(instance_df)
        
        return {"shap_object": shap_values_obj}
    
    
# ----------------------------
# MarketSimulatorV2
# ----------------------------

class MarketSimulatorV2(BaseEnvironment):
    """
    An advanced, standalone quantitative trading environment that supports both
    long and short positions. It includes mechanics for borrowing costs on short sales.
    """
    def __init__(self, market_data: pd.DataFrame, initial_capital: float = 100_000.0, seed: Optional[int] = None, borrowing_rate_daily: float = 0.0005):
        """
        Initializes the trading environment.

        Args:
            market_data (pd.DataFrame): DataFrame with historical market data ('Open', 'High', 'Low', 'Close').
            initial_capital (float): The starting cash balance.
            seed (Optional[int]): Random seed for reproducibility.
            borrowing_rate_daily (float): The daily interest rate charged for holding a short position.
        """
        if not all(k in market_data.columns for k in ['Open', 'High', 'Low', 'Close']):
            raise ValueError("Market data must contain 'Open', 'High', 'Low', 'Close' columns.")
            
        self.market_data = market_data
        self.initial_capital = initial_capital
        self.borrowing_rate = borrowing_rate_daily / 2.0
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        self.state: Dict[str, Any] = {}
        self.current_step = 0
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Resets the environment to its initial state."""
        self.current_step = 0
        self.state = {
            "week": self.current_step, # Retaining 'week' for consistency
            "market_data": self.market_data.iloc[self.current_step].to_dict(),
            "cash": self.initial_capital,
            "shares_held": 0.0,
            "portfolio_value": self.initial_capital,
            "last_profit": 0.0,
        }
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """Returns a copy of the current environment state."""
        return self.state.copy()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Applies an action, advances the environment by one time step, and returns the new state."""
        # 1) Snapshot
        previous_portfolio_value = self.state['portfolio_value']
        current_price = float(self.state['market_data']['Close'])
        action_type = str(action.get('type', 'HOLD')).upper()
        action_amount = float(np.clip(action.get('amount', 1.0), 0.0, 1.0))
    
        # 2) Apply action
        if action_type == 'BUY':
            # Buy with a fraction of cash; if short, this reduces negative shares
            cash_to_spend = self.state['cash'] * action_amount
            shares_to_buy = cash_to_spend / current_price
            self.state['shares_held'] += shares_to_buy
            self.state['cash'] -= cash_to_spend
    
        elif action_type == 'SELL':
            # SELL only reduces existing long positions; no new shorts here
            if self.state['shares_held'] > 0:
                shares_to_sell = self.state['shares_held'] * action_amount
                cash_gained = shares_to_sell * current_price
                self.state['shares_held'] -= shares_to_sell
                self.state['cash'] += cash_gained
            # If no long, SELL does nothing (you can warn/log if desired)
    
        elif action_type == 'SHORT':
            # Open/increase a short using a fraction of equity (portfolio value)
            value_to_short = self.state['portfolio_value'] * action_amount
            if value_to_short > 0:
                shares_to_short = value_to_short / current_price
                self.state['shares_held'] -= shares_to_short  # more negative = larger short
                self.state['cash'] += value_to_short          # short sale proceeds increase cash
    
        # HOLD: do nothing
    
        # 3) Daily borrowing cost for shorts
        if self.state['shares_held'] < 0:
            short_position_value = abs(self.state['shares_held']) * current_price
            cost = short_position_value * self.borrowing_rate
            self.state['cash'] -= cost
    
        # 4) Advance time
        self.current_step += 1
        if self.current_step >= len(self.market_data):
            # End of data: keep last market_data and mark week
            self.state['week'] = self.current_step
            return self.get_state()
    
        # 5) Update state for next day
        new_market_data = self.market_data.iloc[self.current_step].to_dict()
        new_price = float(new_market_data['Close'])
        new_portfolio_value = self.state['cash'] + (self.state['shares_held'] * new_price)
        profit_change = new_portfolio_value - previous_portfolio_value
    
        self.state.update({
            "week": self.current_step,
            "market_data": new_market_data,
            "portfolio_value": new_portfolio_value,
            "last_profit": profit_change,
        })
        return self.get_state()

    



# Legacy, no longer active!
class _LegacySymbolicGuardianV6:
    """
    A unified, non-negotiable safety net for multiple domains, with support
    for validating short-selling actions in the quantitative trading domain.
    Includes auto-clipping of BUY/SHORT amounts to respect max ratios.
    """

    def __init__(
        self,
        # --- E-COMMERCE RULES ---
        max_discount_per_week: float = 0.40, max_price_increase_per_week: float = 0.50,
        min_profit_margin_percentage: float = 0.15, max_price: float = 150.0,
        unit_cost: float = 50.0, ad_absolute_cap: float = 5000.0, ad_increase_cap: float = 1000.0,
        # --- QUANT TRADING RULES (V6) ---
        allow_shorting: bool = True,
        max_position_ratio: float = 0.95,  # For long positions
        max_short_ratio: float = 0.50,     # For short positions
        max_action_amount: float = 1.0
    ):
        """Initializes the Guardian with parameters for ALL supported domains."""
        self.cfg = dict(
            # E-commerce config
            max_dn=max_discount_per_week, max_up=max_price_increase_per_week,
            min_margin=min_profit_margin_percentage, max_price=max_price,
            unit_cost=unit_cost, ad_cap=ad_absolute_cap, ad_increase_cap=ad_increase_cap,
            # Quant config
            allow_shorting=allow_shorting,
            max_pos_ratio=max_position_ratio,
            max_short_ratio=max_short_ratio,
            max_act_amount=max_action_amount
        )

    # ----------------------------
    # E-commerce domain (placeholder)
    # ----------------------------
    def _validate_ecommerce_rules(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return {"is_valid": True, "message": "E-commerce action is valid (placeholder)."}

    # ----------------------------
    # Helpers for quant domain
    # ----------------------------

    def _max_buy_amount(self, cash: float, shares_held: float, price: float) -> float:
        """
        Computes the maximum BUY amount. If already short, this represents the
        amount of cash to use to buy back shares. If long or flat, it respects
        the max_pos_ratio to prevent over-leveraging.
        """
        # --- THE FIX IS HERE ---
        # We REMOVED 'shares_held < 0' from this condition. An agent MUST be
        # able to BUY even if it is short, in order to close its position.
        if price <= 0 or cash < 0:
            return 0.0
        
        # If we are short, any BUY action is valid as it reduces risk.
        # We can use up to 100% of our cash to close the short position.
        if shares_held < 0:
            return 1.0
        # --- END OF FIX ---

        # If we are not short, the original logic to prevent too large
        # long positions still applies.
        target = self.cfg['max_pos_ratio']

        def ratio_for_amt(amt: float) -> float:
            cash_spend = cash * amt
            if price <= 0: return float('inf')
            sh_buy = cash_spend / price
            f_sh = shares_held + sh_buy
            f_cash = cash - cash_spend
            f_pv = f_cash + f_sh * price
            if f_pv <= 0: return float('inf')
            return (f_sh * price) / f_pv

        lo, hi = 0.0, 1.0
        for _ in range(24): # Binary search for precision
            mid = (lo + hi) / 2
            if ratio_for_amt(mid) <= target:
                lo = mid
            else:
                hi = mid
        
        return min(lo, self.cfg['max_act_amount'])

    def _max_short_amount(self, cash: float, shares_held: float, price: float, current_portfolio_value: float) -> float:
        """
        Compute a conservative maximum SHORT amount in [0,1] such that future short ratio
        does not exceed cfg['max_short_ratio'].
        For initial states with only cash: short_ratio ≈ (PV*amt)/PV = amt; clip to target.
        If cash < PV (has long exposure), keep conservative clip based on PV.
        """
        target = self.cfg['max_short_ratio']
        if not self.cfg['allow_shorting'] or price <= 0 or current_portfolio_value <= 0:
            return 0.0

        # Conservative cap: respect target directly relative to PV
        base_cap = target
        # If cash is a fraction of PV, optionally tighten cap
        if cash > 0 and cash < current_portfolio_value:
            # Tighten proportional to cash share (more conservative)
            base_cap = max(0.0, min(target * (cash / current_portfolio_value), target))

        return max(0.0, min(base_cap, self.cfg['max_act_amount']))

    # ----------------------------
    # Quant domain validation
    # ----------------------------
    def _validate_quant_rules(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        action_type = str(action.get('type', 'HOLD')).upper()
        amount = float(action.get('amount', 0.0))

        # Global amount bounds
        if not 0.0 <= amount <= self.cfg['max_act_amount']:
            return {"is_valid": False, "message": f"Rule Violation: Action 'amount' ({amount:.2f}) must be between 0.00 and {self.cfg['max_act_amount']:.2f}."}

        if action_type == 'HOLD':
            return {"is_valid": True, "message": "Action is valid."}

        # Extract portfolio state
        cash = float(state.get('cash', 0.0))
        shares_held = float(state.get('shares_held', 0.0))
        current_price = state.get('market_data', {}).get('Close')

        if current_price is None or current_price <= 0:
            return {"is_valid": False, "message": "State Error: Invalid market price for validation."}

        current_portfolio_value = cash + (shares_held * current_price)
        if current_portfolio_value <= 0:
            # No equity: only HOLD is safe
            return {"is_valid": False, "message": "State Error: Non-positive portfolio value."}

        # BUY validation with auto-clip
        if action_type == 'BUY':
            max_amt = self._max_buy_amount(cash, shares_held, current_price)
            if max_amt <= 0.0:
                return {"is_valid": False, "message": f"Rule Violation: BUY not allowed given max long ratio {self.cfg['max_pos_ratio']:.0%}."}
            if amount > max_amt:
                return {
                    "is_valid": True,
                    "message": f"BUY clipped to {max_amt:.2f} to respect max long ratio {self.cfg['max_pos_ratio']:.0%}.",
                    "adjusted_amount": max_amt
                }
            return {"is_valid": True, "message": "Action is valid."}

        # SELL validation
        if action_type == 'SELL':
            # SELL reduces existing long only; cannot sell if no long
            if shares_held > 0:
                return {"is_valid": True, "message": "Action is valid."}
            return {"is_valid": False, "message": "Rule Violation: No long position to SELL."}

        # SHORT validation with auto-clip
        if action_type == 'SHORT':
            if not self.cfg['allow_shorting']:
                return {"is_valid": False, "message": "Rule Violation: Short-selling is disabled."}
            max_amt = self._max_short_amount(cash, shares_held, current_price, current_portfolio_value)
            if max_amt <= 0.0:
                return {"is_valid": False, "message": "Rule Violation: SHORT not allowed given current cash/equity."}
            if amount > max_amt:
                return {
                    "is_valid": True,
                    "message": f"SHORT clipped to {max_amt:.2f} to respect max short ratio {self.cfg['max_short_ratio']:.0%}.",
                    "adjusted_amount": max_amt
                }
            return {"is_valid": True, "message": "Action is valid."}

        # Unknown action
        return {"is_valid": False, "message": f"Rule Violation: Unknown action type '{action_type}'."}

    # ----------------------------
    # Domain switchboard
    # ----------------------------
    def validate_action(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Route action to correct domain validator."""
        # Minimal domain detection: e-commerce uses price_change/ad_spend keys
        if 'price_change' in action or 'ad_spend' in action:
            return self._validate_ecommerce_rules(action, current_state)
        # Quant trading actions
        elif str(action.get('type', '')).upper() in ['BUY', 'SELL', 'HOLD', 'SHORT']:
            return self._validate_quant_rules(action, current_state)
        # Unknown domain
        return {"is_valid": False, "message": f"Rule Violation: Unrecognized action domain '{action.get('type')}'."}

        
# ----------------------------
# CausalEngineV7_Quant (Final)
# ----------------------------
class CausalEngineV7_Quant:
    """
    Strategic causal reasoning engine for quantitative trading.
    Uses a Causal Forest model trained on historical data to estimate
    the profit impact of BUY, SELL, SHORT, or HOLD actions under given market conditions.
    """

    def __init__(self, data_path: str = "causal_training_data.csv"):
        print("\n--- Initializing CausalEngineV7_Quant ---")
        try:
            self.training_df = pd.read_csv(data_path)
            print(f"✅ Training data loaded successfully from '{data_path}'.")
        except FileNotFoundError:
            print(f"❌ ERROR: Training data not found at '{data_path}'. Please run 'prepare_training_data.py' first.")
            raise

        # Explicit feature set to avoid drift
        self.feature_cols = FEATURE_COLS_DEFAULT
        # Keep only those present in training data
        self.feature_cols = [c for c in self.feature_cols if c in self.training_df.columns]

        # Ensure action_type is numeric directional code
        action_map = {'BUY': 1, 'SHORT': -1, 'SELL': 0, 'HOLD': 0}
        if self.training_df['action_type'].dtype == object:
            self.training_df['action_type'] = self.training_df['action_type'].map(action_map).fillna(0)

        self.model = None
        self._fit_model()
        print("✅ Causal Forest model trained. Engine is ready.")
        print("-----------------------------------------")

    def _fit_model(self):
        """Fit the Causal Forest model on training data."""
        Y = self.training_df['outcome_profit_change']
        T = self.training_df[['action_type', 'action_amount']]
        XW = self.training_df[self.feature_cols]

        self.model = CausalForestDML(
            model_y=GradientBoostingRegressor(random_state=42),
            model_t=MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
            discrete_treatment=False,
            random_state=42,
        )
        self.model.fit(Y, T, X=XW, W=XW)

    def estimate_causal_effect(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Estimate the causal profit impact of a given action in a given context.

        Args:
            action (Dict): e.g., {'type': 'BUY', 'amount': 0.5}
            context (Dict): market features (RSI, SMA, BBands, OHLC)

        Returns:
            float: predicted profit change
        """
        if self.model is None:
            raise RuntimeError("Causal model is not trained.")

        # Prepare context with all required features
        x = {col: context.get(col, np.nan) for col in self.feature_cols}
        X_context = pd.DataFrame([x], columns=self.feature_cols)

        # Control: do-nothing baseline
        T0 = np.array([[0.0, 0.0]])

        # Treatment: directional code + amount
        t = str(action.get('type', 'HOLD')).upper()
        amt = float(action.get('amount', 0.0))
        type_code = 1.0 if t == 'BUY' else (-1.0 if t == 'SHORT' else 0.0)
        T1 = np.array([[type_code, amt]])

        effect = self.model.effect(X_context, T0=T0, T1=T1)
        return float(effect[0])
