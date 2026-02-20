# ecom_multi_agent_benchmark.py
"""
This script orchestrates a competitive multi-agent simulation to benchmark the
performance and strategic behavior of different AI agent personas.

It initializes a competitive market environment (`EcommerceSimulatorV7`) and pits
several AI agents, each with a unique strategic doctrine, against each other
over a simulated period.

The script's main responsibilities include:
1.  Setting up the simulation environment, causal engine, and agents.
2.  Running the main simulation loop week by week.
3.  Collecting detailed historical data on each agent's actions and outcomes.
4.  Generating a comprehensive analysis of the results, including:
    - A summary performance table.
    - A 2x2 dashboard of key metrics over time.
    - Advanced plots for cumulative profit, strategy mapping, and market share.
"""

import os
import sys
import json
import re
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import EcommerceSimulatorV7, CausalEngineV6, SymbolicGuardianV4


# --- Global Configuration & Prompts ---

NUM_AGENTS = 3
NUM_WEEKS = 52

# REFACTOR: Moved from create_competitive_agent_executor for better readability.
HUMAN_PROMPT_TEMPLATE = """
My Agent ID is {agent_id}. Current Week: {week}.
---
My Strategic Doctrine:
{goal}
---
Current Market State (including your rivals' last known positions):
{state_json}
---
Based on all the information above, please proceed with your decision-making process.
"""

# REFACTOR: Moved from create_competitive_agent_executor for better readability.
SYSTEM_PROMPT = """"You are a world-class AI business strategist competing in a market.
Your reasoning process MUST follow this strict workflow:
1.  **Analysis:** Analyze the market state, your competitors, and your specific objective provided in the user prompt.
2.  **Hypothesis Generation:** Formulate THREE diverse and bold hypotheses for actions. These hypotheses MUST NOT be passive (e.g., avoid suggesting `price_change: 0.0` and `ad_spend: 0.0` unless there is an extremely strong strategic reason).
3.  **Validation:** Use the `check_business_rules` tool on EACH of your three hypotheses to see if they are valid.
4.  **Estimation:** For all VALID hypotheses, use the `estimate_profit_impact` tool to predict their long-term value.
5.  **Final Decision:** Compare the estimated outcomes of your valid hypotheses. Choose the one that best aligns with your strategic doctrine (provided in the user prompt).
6.  **Final Output:** Your final answer MUST be a single JSON code block. Do not add any text or explanation before or after the JSON block. Your entire output should be only the JSON.

A perfect example of a final output is:
```json
{{
  "action": {{
    "price_change": -0.10,
    "ad_spend": 1500.0
  }},
  "predicted_profit_change": 25000.0,
  "rationale": "My analysis of the competitor positions and my aggressive growth doctrine indicates that a tactical 10% price drop, supported by a significant ad campaign, will capture market share and yield the highest long-term profit, even if short-term margins are slightly reduced. The causal model predicts a strong positive impact."
}}
"""""


# --- Helper Classes and Functions ---

class StateProvider:
    """A callable class to hold the current state for an agent's tools."""
    def __init__(self):
        self.state = {}

    def __call__(self) -> Dict[str, Any]:
        return self.state

    def update(self, new_state: Dict[str, Any]):
        self.state = new_state


def get_decision_from_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely extracts the action and predicted value from the agent's JSON output
    and returns them in a consistent dictionary format.
    """
    output = response.get('output', '{}')
    m = re.search(r'\{[\s\S]*\}', output)
    try:
        json_string = m.group(0) if m else "{}"
        json_output = json.loads(json_string)

        action = json_output.get('action')
        if not isinstance(action, dict):
            action = {"price_change": 0.0, "ad_spend": 0.0}
        else:
            action.setdefault('price_change', 0.0)
            action.setdefault('ad_spend', 0.0)

        return {
            'action': action,
            'predicted_value': json_output.get('predicted_profit_change', 0.0)
        }
    except (json.JSONDecodeError, AttributeError):
        # CLEANUP: Added tqdm.write for better logging, consistent with benchmark.py
        tqdm.write(f"\n[Warning] Failed to parse agent JSON, defaulting to no-op. Raw output: {output}")
        return {
            "action": {"price_change": 0.0, "ad_spend": 0.0},
            "predicted_value": 0.0
        }


def create_competitive_agent_executor(
    openai_api_key: str,
    state_provider: StateProvider,
    causal_engine: CausalEngineV6
) -> AgentExecutor:
    """Creates a full agent executor with a robust prompt architecture for competition."""
    guardian = SymbolicGuardianV4()

    @tool(description="Checks if a proposed action (price_change, ad_spend) violates any predefined business rules.")
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        action_dict = {"price_change": price_change, "ad_spend": ad_spend}
        return json.dumps(guardian.validate_action(action=action_dict, current_state=state_provider()))

    @tool(description="Estimates the long-term causal impact of an action on a trust-adjusted profit score.")
    def estimate_profit_impact(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        action_dict = {"price_change": price_change, "ad_spend": ad_spend}
        return json.dumps(causal_engine.estimate_causal_effect(action=action_dict, context=state_provider()))

    tools = [check_business_rules, estimate_profit_impact]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=openai_api_key)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


# --- Core Script Functions ---

def setup_competition(openai_api_key: str) -> Tuple:
    """Initializes all components required for the simulation."""
    print(f"--- Setting up a {NUM_AGENTS}-agent, {NUM_WEEKS}-week competitive simulation ---")

    model_data_path = "models/initial_causal_data.pkl"
    if os.path.exists(model_data_path):
        print(f"Deleting old causal data file for a fresh start: {model_data_path}")
        os.remove(model_data_path)

    simulator = EcommerceSimulatorV7(num_agents=NUM_AGENTS, seed=42)
    causal_engine = CausalEngineV6(
        data_path="models/initial_causal_data.pkl",
        force_regenerate=True, 
        num_simulations=2000
    )

    agent_names = [
        "Agent 0: The Aggressive Growth Hacker",
        "Agent 1: The Premium Brand Custodian",
        "Agent 2: The Cautious Market Analyst"
    ]
    agent_goals = [
        """Your strategic doctrine is **Aggressive Growth Hacking**. Your single most important metric is **Total Cumulative Profit**. Be ruthless, data-driven, and opportunistic. Brand Trust is a resource to be spent for significant short-term profit.""",
        """Your strategic doctrine is **Premium Brand Custodianship**. Your primary asset is **Brand Trust**. Sustainable, long-term value is more important than short-term profit. Avoid price wars; your primary tool for growth is advertising.""",
        """Your strategic doctrine is **Cautious Market Analysis**. Your main goal is **Stability and Predictability**. Avoid volatility. Your primary directive is to keep the product price within the stable range of **$90 to $120**."""
    ]

    agents = []
    for i in range(NUM_AGENTS):
        state_provider = StateProvider()
        agent_executor = create_competitive_agent_executor(openai_api_key, state_provider, causal_engine)
        agents.append((agent_executor, state_provider))

    return simulator, agents, agent_names, agent_goals


def run_competition_loop(
    simulator: EcommerceSimulatorV7,
    agents: list,
    agent_names: list,
    agent_goals: list
) -> pd.DataFrame:
    """Executes the main simulation loop and returns the complete history."""
    full_history = []
    with tqdm(total=NUM_WEEKS, desc="Simulating Competitive Weeks") as pbar:
        for week in range(NUM_WEEKS):
            actions_for_turn = []
            current_states = simulator.get_state()['agents']

            for i in range(NUM_AGENTS):
                agent_executor, state_provider = agents[i]
                my_current_state = current_states[i]
                
                competitor_info = [
                    {"agent_id": s['agent_id'], "price": s['price'], "brand_trust": s['brand_trust']}
                    for s in current_states if s['agent_id'] != i
                ]
                state_with_competitors = {**my_current_state, "competitor_info": competitor_info}
                state_provider.update(state_with_competitors)

                try:
                    response = agent_executor.invoke({
                        "agent_id": i,
                        "week": week + 1,
                        "goal": agent_goals[i],
                        "state_json": json.dumps(state_with_competitors, indent=2)
                    })
                    actions_for_turn.append(get_decision_from_response(response))
                except Exception as e:
                    tqdm.write(f"\n[Error] Agent {i} invocation failed. Error: {e}")
                    actions_for_turn.append({"action": {"price_change": 0.0, "ad_spend": 0.0}, "predicted_value": 0.0})

            actual_actions = [d['action'] for d in actions_for_turn]
            new_states = simulator.step(actual_actions)
            
            tqdm.write(f"--- Week {week+1}/{NUM_WEEKS} Results ---")
            for i, state in enumerate(new_states):
                agent_name = agent_names[i]
                action_taken = actions_for_turn[i]['action']
                price_change_pct = action_taken.get('price_change', 0.0) * 100
                ad_spend_val = action_taken.get('ad_spend', 0.0)
                tqdm.write(
                    f"  - {agent_name:<35}: "
                    f"Action(Price Chg: {price_change_pct:+.1f}%, Ad: ${ad_spend_val:<5,.0f}) -> "
                    f"Outcome(Profit: ${state['profit']:<7,.0f}, Mkt Share: {state['market_share']:.1%})"
                )

            for state in new_states:
                state_copy = state.copy()
                state_copy['week'] = week + 1
                full_history.append(state_copy)
            
            pbar.update(1)

    return pd.DataFrame(full_history)


def analyze_and_visualize_results(df: pd.DataFrame, agent_names: List[str]):
    """Analyzes the simulation results and generates all plots and summaries."""
    print("\n--- Competition Finished! Analyzing Results... ---")
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    df.to_csv("multi_agent_results.csv", index=False)
    print("Full simulation results saved to 'multi_agent_results.csv'")
    df['agent'] = df['agent_id'].apply(lambda x: agent_names[x])

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Summary Table ---
    summary_data = []
    for name in agent_names:
        agent_df = df[df['agent'] == name]
        total_profit = agent_df['profit'].sum()
        final_trust = agent_df['brand_trust'].iloc[-1]
        summary_data.append({
            "Agent": name,
            "Total Cumulative Profit": f"${total_profit:,.2f}",
            "Final Brand Trust": f"{final_trust:.3f}"
        })
    summary_df = pd.DataFrame(summary_data)
    sort_key = summary_df['Total Cumulative Profit'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
    summary_df = summary_df.iloc[sort_key.argsort()[::-1]]
    
    print("\n--- COMPETITION SUMMARY (Sorted by Profit) ---")
    print(summary_df.to_string(index=False))

    # --- Main 2x2 Benchmark Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f'{NUM_AGENTS}-Agent Competition Over {NUM_WEEKS} Weeks', fontsize=18)
    metrics = ['profit', 'brand_trust', 'price', 'market_share']
    titles = ['Weekly Profit', 'Brand Trust', 'Price', 'Market Share']
    y_labels = ['Profit ($)', 'Brand Trust Score', 'Price ($)', 'Market Share']
    
    for i, metric in enumerate(metrics):
        ax = axes.flatten()[i]
        sns.lineplot(ax=ax, data=df, x='week', y=metric, hue='agent', linewidth=2.5)
        ax.set_title(titles[i], fontsize=14)
        ax.set_xlabel('Week')
        ax.set_ylabel(y_labels[i])
        ax.legend(title='Agent')
        ax.grid(True)
    
    graph_filename = os.path.join(results_dir, f'multi_agent_main_dashboard_{NUM_WEEKS}weeks.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(graph_filename)
    plt.close(fig)
    print(f"\nMain dashboard graph saved to '{graph_filename}'")

    # --- Advanced Analysis Plots ---
    # 1. Cumulative Profit
    plt.figure(figsize=(12, 8))
    df['cumulative_profit'] = df.groupby('agent')['profit'].cumsum()
    sns.lineplot(data=df, x='week', y='cumulative_profit', hue='agent', linewidth=2.5)
    plt.title('Cumulative Profit Over Time (Who is Winning?)', fontsize=16)
    plt.ylabel('Total Cumulative Profit ($)'); plt.xlabel('Week'); plt.legend(title='Agent'); plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'analysis_cumulative_profit.png'))
    plt.close()
    print("Saved cumulative profit graph.")

    # 2. Strategy Map
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='price', y='weekly_ad_spend', hue='agent', style='agent', s=150, alpha=0.8)
    plt.title('Price vs. Ad Spend Strategy Map', fontsize=16)
    plt.xlabel('Price ($)'); plt.ylabel('Weekly Ad Spend ($)'); plt.legend(title='Agent'); plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'analysis_strategy_map.png'))
    plt.close()
    print("Saved strategy map graph.")
    
    # 3. Market Share Evolution
    market_share_pivot = df.pivot_table(index='week', columns='agent', values='market_share')
    market_share_pivot.plot(kind='area', stacked=True, figsize=(12, 8), alpha=0.7)
    plt.title('Market Share Evolution Over Time', fontsize=16)
    plt.ylabel('Market Share'); plt.xlabel('Week'); plt.ylim(0, 1); plt.legend(title='Agent'); plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'analysis_market_share_area.png'))
    plt.close()
    print("Saved market share evolution graph.")

    # 4. Strategy Distributions
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)
    fig.suptitle('Strategy Distributions (Volatility & Tendency)', fontsize=18)
    sns.boxplot(ax=axes[0], data=df, x='agent', y='price'); axes[0].set_title('Price Distribution')
    sns.boxplot(ax=axes[1], data=df, x='agent', y='weekly_ad_spend'); axes[1].set_title('Ad Spend Distribution')
    sns.boxplot(ax=axes[2], data=df, x='agent', y='profit'); axes[2].set_title('Weekly Profit Distribution')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(results_dir, 'analysis_distributions.png'))
    plt.close(fig)
    print("Saved strategy distribution graphs.")


def main():
    """Main execution function."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        return

    simulator, agents, agent_names, agent_goals = setup_competition(openai_api_key)
    results_df = run_competition_loop(simulator, agents, agent_names, agent_goals)
    analyze_and_visualize_results(results_df, agent_names)


if __name__ == "__main__":
    main()