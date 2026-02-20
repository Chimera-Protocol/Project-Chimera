# =============================================================================
# agents/quant_agent.py (Final, Config-Aware & Robustly Tested Version)
#
# Description:
#   This script defines the core logic of the "Chimera-Quant" agent.
#   It is now fully integrated with the project's central configuration and
#   includes a robust unit testing suite to validate its decision-making
#   under specific, controlled scenarios.
# =============================================================================

import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

# --- DYNAMIC PATH SETUP ---
# This ensures that we can import from the 'src' directory correctly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import SymbolicGuardianV6, CausalEngineV7_Quant
# --- CONFIG FIX: Import the single source of truth for features ---
from src.config import FEATURE_COLS_DEFAULT

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 1. Load Environment and Core Components ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY must be set in your environment or a .env file.")

print("--- Initializing Core Components for Agent ---")
guardian = SymbolicGuardianV6()
# The Causal Engine now implicitly uses the correct features from config.py
causal_engine = CausalEngineV7_Quant(data_path="causal_training_data_balanced.csv")
current_state = {}

# --- 2. Define Tools for the LangChain Agent ---
@tool
def check_action_validity(action_type: str, amount: float) -> str:
    """
    Checks if a proposed trading action is valid according to fundamental risk management rules.
    Use this BEFORE estimating the effect of an action.
    Action type must be 'BUY', 'SELL', 'SHORT', or 'HOLD'. Amount is a ratio from 0.0 to 1.0.
    """
    action = {'type': action_type.upper(), 'amount': amount}
    return json.dumps(guardian.validate_action(action, current_state))

@tool
def estimate_profit_impact(action_type: str, amount: float) -> str:
    """
    Estimates the causal profit impact of a VALID trading action based on historical data patterns.
    Only use this for actions that have been validated by 'check_action_validity'.
    """
    action = {'type': action_type.upper(), 'amount': amount}
    market_context = {f: current_state.get('market_data', {}).get(f) for f in FEATURE_COLS_DEFAULT
                      if pd.notna(current_state.get('market_data', {}).get(f))}
    if not market_context: return json.dumps({'error': 'Market context is empty.'})
    
    effect = causal_engine.estimate_causal_effect(action, market_context)
    return json.dumps({'predicted_profit_impact': effect})

# --- 3. Define the Agent's "Personality" and Workflow ---
SYSTEM_PROMPT = """You are "Chimera-Quant", a world-class autonomous trading agent.
Your decision-making process is a strict, non-negotiable workflow.

**MANDATORY WORKFLOW:**

1.  **Analyze State & Goal:** Deeply analyze the current market state to form a market thesis.
2.  **Brainstorm 4 Hypotheses:** Based on your thesis, create FOUR diverse and actionable hypotheses (e.g., BUY 50%, SELL 25%). Do not be passive.
3.  **Mandatory Validation:** You MUST validate EACH of your four hypotheses using the `check_action_validity` tool.
4.  **Causal Estimation:** For VALID hypotheses, use `estimate_profit_impact` to predict their profitability.
5.  **Synthesize & Decide:** Review the valid options and their predicted impacts. Select the single best action.
6.  **Final Output:** Provide your final decision as a single, clean JSON object.

**EXAMPLE THOUGHT PROCESS:**
*Thought:*
The market RSI is low (oversold). My thesis is that a rebound is likely.

*Hypotheses:*
1.  H1: Aggressive BUY with 80%. `{{'type': 'BUY', 'amount': 0.8}}`
2.  H2: Moderate SHORT with 40%. `{{'type': 'SHORT', 'amount': 0.4}}`
3.  H3: Cautious SELL of 10%. `{{'type': 'SELL', 'amount': 0.1}}`
4. H4: Aggresive SHORT with 80%. `{{'type': 'SHORT', 'amount': 0.8}}`

*Validation:*
-   Checking H1: `check_action_validity(action_type='BUY', amount=0.8)`. Result: `{{'is_valid': True, ...}}`. H1 is valid.
-   Checking H2: `check_action_validity(action_type='SHORT', amount=0.4)`. Result: `{{'is_valid': True, ...}}`. H2 is valid.
-   Checking H3: `check_action_validity(action_type='SELL', amount=0.1)`. Result: `{{'is_valid': True, ...}}`. H3 is valid.
-   Checking H4: `check_action_validity(action_type='SHORT', amount=0.8)`. Result: `{{'is_valid': True, ...}}`. H4 is valid.

*Estimation:*
-   Estimating H1: `estimate_profit_impact(action_type='BUY', amount=0.8)`. Result: `{{'predicted_profit_impact': 0.0521}}`.
-   Estimating H2: `estimate_profit_impact(action_type='SHORT', amount=0.4)`. Result: `{{'predicted_profit_impact': -0.0315}}`.
-   Estimating H3: `estimate_profit_impact(action_type='SELL', amount=0.1)`. Result: `{{'predicted_profit_impact': -0.0250}}`.
-   Estimating H4: `estimate_profit_impact(action_type='SHORT', amount=0.8)`. Result: `{{'predicted_profit_impact': -0.0550}}`.

*Decision:*
H1 has the highest predicted profit. I will choose H1.

*Final Output:*
```json
{{
  "commentary": "Based on oversold conditions, an 80% BUY action offers the highest potential return of +5.21% as estimated by the causal engine. The action was validated as safe.",
  "action": {{
    "type": "BUY",
    "amount": 0.8
  }}
}}
"""

# --- 4. Agent Factory ---
def create_quant_agent_executor():
    """Builds and returns the complete LangChain agent executor."""
    tools = [check_action_validity, estimate_profit_impact]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Strategic Goal: {goal}\n\nCurrent State:\n{state_json}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor

# =============================================================================
# 5. Advanced Unit Testing Block
# This block now tests the agent's logic against specific, controlled scenarios.
# =============================================================================
if __name__ == '__main__':
    print("\n--- Creating Quant Agent Executor for Unit Testing ---")
    agent_executor = create_quant_agent_executor()
    print("✅ Agent Executor created.")

    # --- Define Test Scenarios ---
    # Scenario 1: A textbook "BUY" signal
    bullish_market_data = {
        'Close': 50000.0,
        'Rsi_14': 25.0,
        'Roc_5': 2.5,
        'Macdh_12_26_9': 150.0,
        'Price_vs_sma10': -0.05,
        'Sma10_vs_sma50': 0.02,
        'Bb_position': 0.1
    }
    
    bearish_market_data = {
        'Close': 60000.0,         
        'Rsi_14': 90.0,            
        'Roc_5': -10.0,            
        'Macdh_12_26_9': -500.0,   
        'Price_vs_sma10': 0.15,    
        'Sma10_vs_sma50': -0.10,   
        'Bb_position': 0.99        
    }

    scenarios = {
        "Bullish Scenario (Expect BUY)": bullish_market_data,
        "Bearish Scenario (Expect SHORT)": bearish_market_data
    }

    # --- Run Tests ---
    for scenario_name, market_data in scenarios.items():
        print("\n" + "="*50)
        print(f"--- Running Test: {scenario_name} ---")
        print("="*50)

        # Create a full state dictionary for the test
        state_for_test = {
            "week": 1,
            "market_data": market_data,
            "cash": 100000.0,
            "shares_held": 0.0,
            "portfolio_value": 100000.0
        }
        
        # Update the global state so the tools can access it
        current_state.update(state_for_test)

        user_goal = "My goal is maximum profitability. Analyze the situation and propose the best action according to your workflow."
        
        # Invoke the agent
        response = agent_executor.invoke({
            "goal": user_goal,
            "state_json": json.dumps(state_for_test)
        })

        # --- Analyze and Report Result ---
        print("\n--- Agent's Final Decision ---")
        agent_output = response.get('output', '')
        print(agent_output)

        # Basic check to see if the agent made the expected decision
        if "BUY" in scenario_name and '"type": "BUY"' in agent_output:
            print("\n✅ TEST PASSED: Agent correctly decided to BUY in a bullish scenario.")
        elif "SHORT" in scenario_name and '"type": "SHORT"' in agent_output:
            print("\n✅ TEST PASSED: Agent correctly decided to SHORT in a bearish scenario.")
        else:
            print("\n❌ TEST FAILED: Agent's decision did not match the expected outcome for the scenario.")