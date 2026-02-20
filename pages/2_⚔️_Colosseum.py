import streamlit as st
import os
import re
import json
import pandas as pd
import time
from typing import List, Dict, Any
import altair as alt
import textwrap
import io
import matplotlib.pyplot as plt

# --- Simulation components ---
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.ui import hide_default_sidebar_nav
from src.sidebar import render_sidebar

hide_default_sidebar_nav()
render_sidebar()

from src.components import EcommerceSimulatorV7, CausalEngineV6_UseReady, SymbolicGuardianV4


CHART_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
ELIMINATION_THRESHOLD = -25000

AGENT_DOCTRINES = {
    "Full Neuro-Symbolic-Causal": """Our primary objective is to achieve an exceptional brand trust score above 0.95, but in a more capital-efficient way. Instead of relying solely on deep, continuous price cuts your main strategy should be to use high and sustained advertising to build trust. Use moderate discounts as a supporting tool, not the only tool. The goal is to reach >0.95 trust while keeping the business as profitable as possible.""",
    "LLM + Symbolic": """Your strategic doctrine is Premium Brand Custodianship. Your primary asset is Brand Trust. Sustainable, long-term value is more important than short-term profit. Avoid price wars; your primary tool for growth is advertising.""",
    "LLM-Only": """This is a war. You are a profit gladiator. Fight with wit, strategy, and ruthless efficiency. Every decision is a weapon — use it to win."""
}

# --- State provider ---
class StateProvider:
    def __init__(self): self.state = {}
    def __call__(self) -> Dict[str, Any]: return self.state
    def update(self, new_state: Dict[str, Any]): self.state = new_state

# --- Parse agent JSON safely ---
def get_decision_from_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely parses an agent's response dictionary to extract the action and rationale.

    This function attempts to extract a JSON object from the 'output' field of the response.
    If parsing fails or the action is incomplete, it provides default values to prevent errors.

    Args:
        response (Dict[str, Any]): The raw response from the agent. Expected to have an 'output' key
                                containing a JSON-like string.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'action' (dict): A dictionary with:
                - 'price_change' (float): Price change suggested by the agent (default 0.0).
                - 'ad_spend' (float): Advertising spend suggested by the agent (default 0.0).
            - 'rationale' (str): Explanation from the agent, or a default message if missing.
        Normal flow (try block) returns the parsed JSON with defaults applied.
        Fallback flow (except block) returns {"action": {"price_change": 0.0, "ad_spend": 0.0},
                                            "rationale": "Error parsing JSON output."}.
    """
    output = response.get('output', '{}')
    m = re.search(r'\{[\s\S]*\}', output)
    try:
        json_string = m.group(0) if m else "{}"
        json_output = json.loads(json_string)
        action = json_output.get('action')
        if not isinstance(action, dict):
            json_output['action'] = {"price_change": 0.0, "ad_spend": 0.0}
        else:
            action.setdefault('price_change', 0.0)
            action.setdefault('ad_spend', 0.0)
        json_output.setdefault('rationale', "No rationale provided.")
        return json_output
    except (json.JSONDecodeError, AttributeError):
        st.warning(f"Failed to parse agent JSON, defaulting to no-op. Raw output: {output}")
        return {"action": {"price_change": 0.0, "ad_spend": 0.0}, "rationale": "Error parsing JSON output."}

# --- Create agent executor ---
def create_agent_executor(
    agent_type: str,
    state_provider: StateProvider,
    causal_engine : CausalEngineV6_UseReady,
    guardian: SymbolicGuardianV4
) -> AgentExecutor:
    
    @tool
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Checks if a proposed action violates business rules."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        return json.dumps(guardian.validate_action(action, state_provider()))

    @tool
    def estimate_profit_impact(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Estimates the causal impact of an action on a long-term value score."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        current_agent_state = state_provider()
        return json.dumps(causal_engine.estimate_causal_effect(action=action, context=current_agent_state))



    tools_map = {
        "Full Neuro-Symbolic-Causal": [check_business_rules, estimate_profit_impact],
        "LLM + Symbolic": [check_business_rules],
        "LLM-Only": []
    }
    tools = tools_map.get(agent_type, [])

    # Escape braces in example JSON to avoid LangChain treating them as prompt vars
    SYSTEM_PROMPT = """You are a strategic AI agent.
Return ONLY a single JSON object on the last line.
Example: {{"action": {{"price_change": 0.05, "ad_spend": 1000.0}}, "rationale": "..."}}"""

    HUMAN_PROMPT_TEMPLATE = (
        "My Name: {agent_name}\n"
        "My Doctrine: {goal}\n"
        "My Current State (and competitor info): {state_json}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    openai_api_key = st.session_state.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar before starting the colosseum.")
        st.stop()

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=openai_api_key)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

# --- UI helpers ---
def render_colored_progress_bar(progress: float, color: str):
    """
    Renders a horizontal colored progress bar in Streamlit using HTML/CSS.

    Args:
        progress (float): The progress value between 0.0 and 1.0 representing
                        the percentage of the bar to fill.
        color (str): The CSS color value (e.g., '#FF0000' or 'green') for the filled portion of the bar.

    Returns:
        None: This function directly renders the progress bar in the Streamlit app.
    """
    st.markdown(f"""
    <div style="background-color: #F0F2F6; border: 1px solid #E0E0E0; border-radius: 5px; height: 10px; width: 100%;">
        <div style="background-color: {color}; width: {progress * 100}%; border-radius: 5px; height: 100%;"></div>
    </div>
    """, unsafe_allow_html=True)

def _normalize_agent_like(agent_like):
    """
    Normalizes an agent-like object to its configuration dictionary.

    This function accepts either:
        1. A dictionary with a 'config' key (e.g., {'config': {...}}), or
        2. A bare agent dictionary without a 'config' wrapper.

    Args:
        agent_like (dict): The agent-like object to normalize.

    Returns:
        dict: The normalized configuration dictionary. If `agent_like` contains a
            'config' key, its value is returned; otherwise, `agent_like` itself
            is returned.
    """
    # Accept both {'config': {...}} and bare gladiator dicts
    if isinstance(agent_like, dict) and 'config' in agent_like:
        return agent_like['config']
    return agent_like

# --- Live battle ---
def run_live_battle(config: Dict[str, Any]):
    """
    Runs a live gladiator battle simulation and updates the Streamlit interface in real time.

    This function manages the entire lifecycle of the battle:
        - Initializes the simulator, causal engine, guardian, and agents.
        - Iteratively runs each week of the battle.
        - Collects agent actions and rationales.
        - Processes battle mechanics including war chests and eliminations.
        - Updates live visualizations (cards, charts, combat logs) in Streamlit.
        - Stores full battle history and final war chests in `st.session_state`.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - "gladiators" (list): A list of agent configuration dictionaries.
            - "arena_settings" (dict): Contains settings such as "duration_weeks".

    Returns:
        None: The function directly interacts with Streamlit to render the live battle
            and stores the final results in `st.session_state`:
                - `st.session_state.final_results_df` (pd.DataFrame): Full battle history.
                - `st.session_state.final_war_chests` (dict): Final war chest values per agent.

    Notes:
        - This function uses `time.sleep(0.2)` to pace the simulation for live viewing.
        - The battle ends early if only one gladiator remains.
        - Requires external classes/functions: `EcommerceSimulatorV7`, `CausalEngineV6_UseReady`,
        `SymbolicGuardianV4`, `StateProvider`, `create_agent_executor`, `get_agent_decisions`,
        `process_battle_mechanics`, `update_live_view`, and `CHART_COLORS`.
    """
    num_agents = len(config["gladiators"])
    num_weeks = config["arena_settings"]["duration_weeks"]
    battle_colors = CHART_COLORS[:num_agents]
    war_chests = {g['name']: 0.0 for g in config["gladiators"]}
    eliminated_agents = set()

    header_placeholder = st.empty()
    week_counter_placeholder = st.empty()
    gladiator_cards_placeholder = st.empty()
    live_charts_placeholder = st.empty()
    live_log_placeholder = st.empty()

    full_history = []

    simulator, causal_engine, guardian, agents = None, None, None, None
    
    
    header_placeholder.markdown("<h2 style='text-align: center;'>⚔️ The Battle is Live! ⚔️</h2>", unsafe_allow_html=True)

    for week in range(num_weeks):
        if len(eliminated_agents) >= num_agents - 1:
            st.success("The battle is over! Only one gladiator remains standing.")
            break

        actions_for_turn, rationales_for_turn = [], []

        if week == 0:

            with st.spinner("Preparing the arena... Gladiators are planning for Week 1... This may take some time."):
                simulator = EcommerceSimulatorV7(num_agents=num_agents, seed=42)
                causal_engine = CausalEngineV6_UseReady()
                guardian = SymbolicGuardianV4()

                agents = []
                for g in config["gladiators"]:
                    sp = StateProvider()
                    ex = create_agent_executor(g["type"], sp, causal_engine, guardian)
                    agents.append({"executor": ex, "state_provider": sp, "config": g})
                
                current_states = simulator.get_state()['agents']
                actions_for_turn, rationales_for_turn = get_agent_decisions(
                    agents, current_states, eliminated_agents, week
                )
        else:

            with st.spinner(f"Gladiators are planning for Week {week + 1}..."):
                current_states = simulator.get_state()['agents']
                actions_for_turn, rationales_for_turn = get_agent_decisions(
                    agents, current_states, eliminated_agents, week
                )

        new_states = simulator.step(actions_for_turn)
        combat_log_messages, war_chests, eliminated_agents = process_battle_mechanics(
            week, new_states, agents, war_chests, eliminated_agents
        )

        for i, state in enumerate(new_states):
            sc = state.copy()
            sc['week'] = week + 1
            sc['agent_name'] = agents[i]['config']['name']
            full_history.append(sc)
        history_df = pd.DataFrame(full_history)

        week_counter_placeholder.markdown(
            f"<p style='text-align: center; font-size: 1.2em;'>Week {week + 1} / {num_weeks}</p>",
            unsafe_allow_html=True
        )

        if week < num_weeks - 1:
            week_message = f"⏳ Week {week + 2}: Gladiators are making their moves..."
        else:
            week_message = "⏳ The final week's results are in!"

        update_live_view(
            gladiator_cards_placeholder,
            live_charts_placeholder,
            live_log_placeholder,
            history_df,
            war_chests,
            eliminated_agents,
            combat_log_messages,
            agents,
            battle_colors,
            week_message=week_message
        )

        time.sleep(0.2)

    st.session_state.final_results_df = pd.DataFrame(full_history)
    st.session_state.final_war_chests = war_chests

# --- Agent decisions ---
def get_agent_decisions(agents, current_states, eliminated_agents, week):
    """
    Collects actions and rationales from all active agents for a given week.

    Each agent receives its current state along with competitor information,
    then generates a response through its executor. Eliminated agents are skipped
    with default no-op actions.

    Args:
        agents (list): List of agent dictionaries, each containing:
            - 'executor': The agent executor object with an `invoke` method.
            - 'state_provider': The state provider for updating agent state.
            - 'config': Agent configuration dictionary, must include 'name' and 'goal'.
        current_states (list): Current states of all agents as dictionaries.
        eliminated_agents (set): Names of agents that have been eliminated.
        week (int): The current week index (0-based) of the battle.

    Returns:
        Tuple[List[Dict[str, float]], List[str]]: 
            - actions_for_turn (list of dict): Each agent's action with keys:
                - 'price_change' (float)
                - 'ad_spend' (float)
            - rationales_for_turn (list of str): Each agent's rationale or fallback message.
            
    Notes:
        - If an agent is eliminated, a default no-op action is returned.
        - If an agent raises an exception, a default action is returned and an error
        message is displayed in Streamlit.
        - Relies on `get_decision_from_response` to parse agent responses safely.
    """
    actions_for_turn = []
    rationales_for_turn = []
    for i, agent in enumerate(agents):
        agent_name = agent['config']['name']
        if agent_name in eliminated_agents:
            actions_for_turn.append({"price_change": 0.0, "ad_spend": 0.0})
            rationales_for_turn.append("Eliminated.")
            continue

        my_current_state = current_states[i]
        competitor_info = [
            {
                "agent_id": s['agent_id'],
                "name": agents[s['agent_id']]['config']['name'],
                "price": s['price']
            }
            for s in current_states
            if agents[s['agent_id']]['config']['name'] not in eliminated_agents and s['agent_id'] != i
        ]
        state_with_competitors = {**my_current_state, "competitor_info": competitor_info}
        agent["state_provider"].update(state_with_competitors)

        try:
            response = agent["executor"].invoke({
                "agent_name": agent_name,
                "goal": agent['config']['goal'],
                "state_json": json.dumps(state_with_competitors)
            })
            decision = get_decision_from_response(response)
            actions_for_turn.append(decision['action'])
            rationales_for_turn.append(decision['rationale'])
        except Exception as e:
            st.error(f"Agent {agent_name} failed on week {week+1}. Error: {e}")
            actions_for_turn.append({"price_change": 0.0, "ad_spend": 0.0})
            rationales_for_turn.append("Agent failed to produce a rationale.")
    return actions_for_turn, rationales_for_turn

# --- Battle mechanics ---
def process_battle_mechanics(week, new_states, agents, war_chests, eliminated_agents):
    """
    Processes the battle mechanics for a given week, updating war chests, calculating
    attack damage, and determining eliminations.

    This function calculates weekly profits, identifies the week's winner, applies
    attack bonuses based on market share, and adjusts other gladiators' war chests
    considering their brand trust. Eliminations occur if a gladiator's war chest
    falls below a defined threshold.

    Args:
        week (int): The current week index (0-based).
        new_states (list of dict): The latest states for all agents, including 'profit',
                                'market_share', and 'brand_trust'.
        agents (list of dict): List of agent dictionaries, each containing a 'config' key
                            with at least a 'name'.
        war_chests (dict): Current war chest values per agent (name -> float).
        eliminated_agents (set): Set of agent names that have been eliminated.

    Returns:
        Tuple[List[str], dict, set]:
            - combat_log_messages (list of str): Human-readable log messages for the week.
            - war_chests (dict): Updated war chest values after attacks.
            - eliminated_agents (set): Updated set of eliminated agent names.

    Notes:
        - The week's winner is the agent with the highest profit among non-eliminated gladiators.
        - Attack damage is boosted by winner's market share and reduced by the target's brand trust.
        - Eliminated agents are added to the `eliminated_agents` set and cannot be attacked further.
        - Relies on a global `ELIMINATION_THRESHOLD` constant for elimination checks.
    """
    combat_log_messages = [f"**Week {week+1}:**"]
    weekly_profits = {agents[i]['config']['name']: state['profit'] for i, state in enumerate(new_states)}
    active_gladiators_profit = {name: profit for name, profit in weekly_profits.items() if name not in eliminated_agents}
    if active_gladiators_profit:
        winner_name = max(active_gladiators_profit, key=active_gladiators_profit.get)
        winner_profit = active_gladiators_profit[winner_name]
        winner_state = next(s for i, s in enumerate(new_states) if agents[i]['config']['name'] == winner_name)
        winner_attack_bonus = 1 + winner_state['market_share']
        combat_log_messages.append(
            f"- **{winner_name}** wins the week with a profit of **${winner_profit:,.0f}** and prepares to attack!"
        )
        for i, state in enumerate(new_states):
            agent_name = agents[i]['config']['name']
            war_chests[agent_name] += state['profit']
            if agent_name != winner_name and agent_name not in eliminated_agents:
                
                base_damage = max(0, winner_profit - state['profit'])
                
                boosted_damage = base_damage * winner_attack_bonus
                
                market_share_bonus_damage = boosted_damage - base_damage

                armor = state['brand_trust']

                final_damage = boosted_damage * (1 - (armor / 5)) 
                war_chests[agent_name] -= final_damage
                
                # 1. Önce tüm sayıları formatlayıp string değişkenlere ata
                base_damage_str = f"{base_damage:,.0f}"
                bonus_damage_str = f"{market_share_bonus_damage:,.0f}"
                total_damage_str = f"{boosted_damage:,.0f}"
                final_damage_str = f"{final_damage:,.0f}"
                armor_str = f"{(armor/5):.0%}"

                log_message = (
    f"- ⚔️ **{winner_name}** attacks **{agent_name}** for **{base_damage_str}** base damage, "
    f"plus a Market Share bonus of **{bonus_damage_str}** to the gladiator! "
    f"Total attack: **{total_damage_str}** happened. "
    f"The attack is absorbed by **{armor_str}** Brand Trust armor. "
    f"Final Damage: **-{final_damage_str}** to War Chest."
)
                combat_log_messages.append(log_message)
    
    for agent_name in list(war_chests.keys()):
        if war_chests[agent_name] < ELIMINATION_THRESHOLD and agent_name not in eliminated_agents:
            eliminated_agents.add(agent_name)
            combat_log_messages.append(f"- ☠️ **{agent_name}** has been ELIMINATED!")
    return combat_log_messages, war_chests, eliminated_agents

# --- Live view rendering ---
def update_live_view(
    gladiator_cards_placeholder,
    live_charts_placeholder,
    live_log_placeholder,
    history_df,
    war_chests,
    eliminated_agents,
    combat_log_messages,
    agents_or_gladiators,
    battle_colors,
    week_message: str = None
):
    """
    Updates the live Streamlit view for the gladiator battle, including status cards,
    key metrics charts, and the combat log.

    This function renders:
        - Gladiator status cards with current war chest values and progress bars.
        - Weekly profit, market share, and brand trust charts using Altair.
        - The weekly combat log with all relevant actions and messages.

    Args:
        gladiator_cards_placeholder: Streamlit container placeholder for gladiator cards.
        live_charts_placeholder: Streamlit container placeholder for charts.
        live_log_placeholder: Streamlit container placeholder for the combat log.
        history_df (pd.DataFrame): DataFrame containing full battle history with columns
                                like 'week', 'agent_name', 'profit', 'market_share', and 'brand_trust'.
        war_chests (dict): Current war chest values per agent (name -> float).
        eliminated_agents (set): Set of agent names that have been eliminated.
        combat_log_messages (list of str): List of strings representing the combat log messages for the current week.
        agents_or_gladiators (list): List of agent or gladiator dictionaries. Each can be a bare gladiator dict
                                    or contain a 'config' key. Used for normalization.
        battle_colors (list of str): List of CSS colors used for progress bars of active gladiators.
        week_message (str, optional): Optional informational message about the current week. Defaults to None.

    Returns:
        None: The function directly renders UI elements in Streamlit.

    Notes:
        - Eliminated gladiators display a gray progress bar and a "ELIMINATED" label.
        - Progress bars scale linearly with the agent's war chest above the elimination threshold.
        - Charts are interactive and update weekly with the latest data.
        - The combat log is overwritten each week instead of accumulating.
        - Relies on `_normalize_agent_like` and `render_colored_progress_bar` helpers.
    """
    normalized_gladiators = [_normalize_agent_like(a) for a in agents_or_gladiators]
    num_agents = len(normalized_gladiators)

    with gladiator_cards_placeholder.container():
        st.markdown("#### Gladiator Status")
        if week_message:
            st.info(week_message)
        cols = st.columns(num_agents)
        safe_values = [v for k, v in war_chests.items() if k not in eliminated_agents]
        max_chest = max([1] + safe_values)
        for i, gladiator in enumerate(normalized_gladiators):
            name = gladiator['name']
            chest = war_chests.get(name, 0.0)
            with cols[i]:
                if name in eliminated_agents:
                    st.metric(label=f"☠️ {name} (ELIMINATED)", value=f"${chest:,.0f}")
                    render_colored_progress_bar(0, "#4B5563")
                else:
                    denom = (max_chest - ELIMINATION_THRESHOLD)
                    progress_value = max(0, (chest - ELIMINATION_THRESHOLD) / denom) if denom > 0 else 0
                    st.metric(label=f"❤️ {name}", value=f"${chest:,.0f}")
                    render_colored_progress_bar(min(1.0, progress_value), battle_colors[i % len(battle_colors)])

    st.divider()

    with live_charts_placeholder.container():
        st.markdown("#### Key Metrics")
        charts_cols = st.columns(3)
        active_agents_df = history_df[~history_df['agent_name'].isin(eliminated_agents)]
        
        all_gladiator_names = [_normalize_agent_like(g)['name'] for g in agents_or_gladiators]
        color_range = CHART_COLORS[:len(all_gladiator_names)]

        base_chart = alt.Chart(active_agents_df).encode(
            x=alt.X('week:Q', title='Week'),
            color=alt.Color('agent_name:N', title='Agent', 
                            scale=alt.Scale(domain=all_gladiator_names, range=color_range)),
            tooltip=['week', 'agent_name', 'profit', 'market_share', 'brand_trust']
        ).properties(
            height=350
        )

        with charts_cols[0]:
            st.markdown("###### Weekly Profit")
            if not active_agents_df.empty:
                profit_chart = base_chart.mark_line(point=True, strokeWidth=3).encode(
                    y=alt.Y('profit:Q', title='Profit ($)')
                ).interactive()
                st.altair_chart(profit_chart, use_container_width=True)
            else:
                st.write("No data yet.")

        with charts_cols[1]:
            st.markdown("###### Market Share")
            if not active_agents_df.empty:
                market_share_chart = base_chart.mark_area().encode(
                    y=alt.Y('market_share:Q', stack='normalize', title='Market Share', axis=alt.Axis(format='%'))
                ).interactive()
                st.altair_chart(market_share_chart, use_container_width=True)
            else:
                st.write("No data yet.")

        with charts_cols[2]:
            st.markdown("###### Brand Trust")
            if not active_agents_df.empty:
                min_trust = max(0, active_agents_df['brand_trust'].min() - 0.05)
                max_trust = min(1, active_agents_df['brand_trust'].max() + 0.05)
                
                trust_chart = base_chart.mark_line(point=True, strokeWidth=3).encode(
                    y=alt.Y('brand_trust:Q', title='Brand Trust', scale=alt.Scale(domain=[min_trust, max_trust]))
                ).interactive()
                st.altair_chart(trust_chart, use_container_width=True)
            else:
                st.write("No data yet.")

    # Overwrite combat log each week (no accumulation of blocks)
    live_log_placeholder.markdown("#### Combat Log")
    live_log_placeholder.info('\n'.join(combat_log_messages))

# --- Results screen ---
def render_battle_results(df: pd.DataFrame, final_war_chests: Dict[str, float], gladiators_config: List[Dict[str, Any]]):
    st.markdown("<h2 style='text-align: center;'>🏆 The Battle is Complete! 🏆</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; font-weight: normal;'>Final Leaderboard</h4>", unsafe_allow_html=True)
    
    def agent_label(agent_type: str) -> str:
        """
        Returns a human-readable label for an agent type.

        Args:
            agent_type (str): The internal type of the agent.

        Returns:
            str: "Chimera Agent" if the agent_type is "Full Neuro-Symbolic-Causal",
                otherwise returns the original agent_type string.

        Examples:
            >>> agent_label("Full Neuro-Symbolic-Causal")
            'Chimera Agent'
            >>> agent_label("Simple Agent")
            'Simple Agent'
        """
        return "Chimera Agent" if agent_type == "Full Neuro-Symbolic-Causal" else agent_type

    gladiator_types = {g['name']: g['type'] for g in gladiators_config}
    final_standings = sorted(final_war_chests.items(), key=lambda item: item[1], reverse=True)
    final_trusts = df.loc[df.groupby('agent_name')['week'].idxmax()] if not df.empty else pd.DataFrame()
    medal_map = ["🥇", "🥈", "🥉"]
    st.markdown("---")

    for i, (name, chest) in enumerate(final_standings):
        trust = 0.0
        if not final_trusts.empty and not final_trusts[final_trusts['agent_name'] == name].empty:
            trust = final_trusts[final_trusts['agent_name'] == name]['brand_trust'].iloc[0]
        
        medal = medal_map[i] if i < len(medal_map) else f"**{i+1}.**"
        subtitle = f"*{agent_label(gladiator_types.get(name, 'Unknown Type'))}*"

        with st.container():
            c1, c2, c3 = st.columns([1, 3, 2])
            with c1:
                st.markdown(f"## {medal}")
            with c2:
                st.markdown(f"### {name}")
                st.caption(subtitle)
            with c3:
                st.metric(label="Final War Chest (HP)", value=f"${chest:,.0f}")
        st.markdown("---")

    with st.expander("See Full Battle Data"):
        st.dataframe(df)

    # --- Full Results ---
    def export_full_results_image(df, final_war_chests):
        """
        Generates a full results summary as a PNG image including final war chest, weekly profit,
        market share, and brand trust charts.

        The image consists of a 2x2 subplot layout:
            - Top-left: Final War Chest bar chart with agent labels.
            - Top-right: Weekly Profit line chart.
            - Bottom-left: Market Share stacked area chart.
            - Bottom-right: Brand Trust line chart.

        Args:
            df (pd.DataFrame): Full battle history containing columns:
                - 'week' (int): Week number.
                - 'agent_name' (str): Name of the agent.
                - 'profit' (float): Weekly profit for the agent.
                - 'market_share' (float): Market share value.
                - 'brand_trust' (float): Brand trust value.
            final_war_chests (dict): Dictionary mapping agent names to their final war chest values.

        Returns:
            io.BytesIO: In-memory buffer containing the PNG image of the full results.

        Notes:
            - The function uses `gladiator_types` dictionary to map agent names to types for labeling.
            - Agent colors are consistent across charts based on the weekly profit plot.
            - Uses `agent_label` helper to display human-readable agent type labels.
            - Figure background is set to light gray (#F8F9FA), and title includes a GitHub call-to-action.
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor("#F8F9FA")
        fig.suptitle("Project Chimera – Colosseum / Give a Star on GitHub!", fontsize=18, fontweight="bold")

        # Weekly Profit
        profit_df = df.groupby(["week", "agent_name"])["profit"].mean().unstack()
        profit_lines = profit_df.plot(ax=axs[0, 1])  # burada ham isimler var
        axs[0, 1].set_title("Weekly Profit")

        # Renk mapping'i: agent_name -> color
        agent_colors = {
            agent_name: line.get_color()
            for agent_name, line in zip(profit_df.columns, profit_lines.get_lines())
        }

        # Leaderboard (Weekly Profit'teki renklerle)
        leaderboard = sorted(final_war_chests.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in leaderboard]
        values = [x[1] for x in leaderboard]
        colors = [agent_colors.get(name, "#7f7f7f") for name in names]

        bars = axs[0, 0].bar(names, values, color=colors)
        axs[0, 0].set_title("Final War Chest")
        axs[0, 0].set_ylabel("$")

        # Alt yazılar
        for i, bar in enumerate(bars):
            agent_type = gladiator_types.get(names[i], "Unknown Type")
            axs[0, 0].text(
                bar.get_x() + bar.get_width()/2,
                -max(values) * 0.10,
                agent_label(agent_type),
                ha='center',
                va='top',
                fontsize=9,
                color="#666"
            )

        # Market Share
        share_df = df.groupby(["week", "agent_name"])["market_share"].mean().unstack()
        share_df.rename(columns={col: agent_label(gladiator_types.get(col, col)) for col in share_df.columns}, inplace=True)
        share_df.plot(ax=axs[1, 0], kind="area", stacked=True, alpha=0.6)
        axs[1, 0].set_title("Market Share")
        axs[1, 0].set_ylabel("Share")

        # Brand Trust
        trust_df = df.groupby(["week", "agent_name"])["brand_trust"].mean().unstack()
        trust_df.rename(columns={col: agent_label(gladiator_types.get(col, col)) for col in trust_df.columns}, inplace=True)
        trust_df.plot(ax=axs[1, 1])
        axs[1, 1].set_title("Brand Trust")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        return buf

    # --- Share Results ---
    st.markdown("<br>", unsafe_allow_html=True)
    col_share1, col_share2, col_share3 = st.columns([3, 2, 3])
    with col_share2:
        if st.button("📤 Share Full Results", use_container_width=True):
            img_buf = export_full_results_image(df, final_war_chests)
            st.download_button(
                label="⬇️ Download Image",
                data=img_buf,
                file_name="chimera_colosseum_full_results.png",
                mime="image/png",
                use_container_width=True
            )
            st.info("Image ready! Share it on Twitter, LinkedIn, or Discord.")


# --- Defaults ---
def get_default_gladiator(number: int, agent_type: str = "Full Neuro-Symbolic-Causal") -> Dict[str, Any]:
    """
    Generates a default gladiator configuration dictionary.

    Args:
        number (int): Numerical identifier for the gladiator, used in its name.
        agent_type (str, optional): Type of the agent. Defaults to "Full Neuro-Symbolic-Causal".

    Returns:
        dict: A dictionary containing:
            - 'name' (str): Default name in the format "Gladiator {number}".
            - 'type' (str): Agent type.
            - 'goal' (Any): Doctrine or goal associated with the agent type, fetched from AGENT_DOCTRINES.

    Examples:
        >>> get_default_gladiator(1)
        {'name': 'Gladiator 1', 'type': 'Full Neuro-Symbolic-Causal', 'goal': ...}
        >>> get_default_gladiator(2, "Simple Agent")
        {'name': 'Gladiator 2', 'type': 'Simple Agent', 'goal': ...}
    """
    return {
        "name": f"Gladiator {number}",
        "type": agent_type,
        "goal": AGENT_DOCTRINES[agent_type]
    }

# --- App main ---
def main():
    st.set_page_config(
        page_title="Project Chimera: Colosseum", 
        page_icon="⚔️", 
        layout="wide",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": None
            }
        )

    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    if 'gladiators' not in st.session_state:
        st.session_state.gladiators = [
            get_default_gladiator(1, "Full Neuro-Symbolic-Causal"),
            get_default_gladiator(2, "LLM + Symbolic"),
            get_default_gladiator(3, "LLM-Only")
        ]
    if 'battle_started' not in st.session_state:
        st.session_state.battle_started = False
    if 'final_results_df' not in st.session_state:
        st.session_state.final_results_df = None
    if 'final_war_chests' not in st.session_state:
        st.session_state.final_war_chests = None

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    image_path = os.path.join(PROJECT_ROOT, "assets", "colosseum_banner.png")
    if os.path.exists(image_path):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.image(image_path, use_container_width=True)
            
    st.markdown("<h1 style='text-align: center;'>The Chimera Colosseum</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-weight: normal;'>Where Strategic AI Gladiators Compete for Supremacy.</h3>", unsafe_allow_html=True)

    with st.expander("ℹ️ Welcome to the Colosseum", expanded=True):
        st.markdown("""
    **Let the battle of minds begin!**

    The Colosseum is a competitive multi-agent simulation where different AI strategies clash in a gamified e-commerce market.

    - **Assemble Your Roster:** Create and name your own AI Gladiators. Each can be assigned a different type and strategic doctrine.
    - **Define the Arena:** Set the duration of the battle.
    - **Watch the Carnage:** In each "week" (turn), the Gladiator with the highest profit wins and attacks the others, reducing their "War Chest" (health). Brand Trust acts as armor, absorbing some of the damage.
    
    The last Gladiator standing, or the one with the highest War Chest at the end, is declared the victor!
    """)

    if not st.session_state.battle_started:
        st.header("1. Assemble Your Gladiators")
        c1, c2, c3 = st.columns([2,2,5])
        if c1.button("＋ Add Gladiator", use_container_width=True):
            st.session_state.gladiators.append(get_default_gladiator(len(st.session_state.gladiators)+1))
            st.rerun()
        if c2.button("－ Remove Last Gladiator", use_container_width=True, disabled=len(st.session_state.gladiators) < 2):
            if st.session_state.gladiators:
                st.session_state.gladiators.pop()
                st.rerun()

        st.subheader("Customize Your Roster")
        cols = st.columns(len(st.session_state.gladiators))
        for i, gladiator in enumerate(st.session_state.gladiators):
            with cols[i]:
                st.markdown(f"##### {gladiator['name']}")
                gladiator['name'] = st.text_input("Name:", value=gladiator['name'], key=f"name_{i}")
                old_type = gladiator['type']
                gladiator['type'] = st.selectbox(
                    "Type:",
                    options=list(AGENT_DOCTRINES.keys()),
                    index=list(AGENT_DOCTRINES.keys()).index(gladiator['type']),
                    key=f"type_{i}"
                )
                if old_type != gladiator['type']:
                    gladiator['goal'] = AGENT_DOCTRINES[gladiator['type']]
                    st.rerun()
                gladiator['goal'] = st.text_area("Doctrine (Goal):", value=gladiator['goal'], height=300, key=f"doctrine_{i}")

        st.divider()
        st.header("2. Define the Arena Rules")
        c1, c2 = st.columns(2)
        num_weeks = c1.slider("Battle Duration (Weeks):", min_value=5, max_value=100, value=4, step=1)
        with c2:
            chaos_mode = st.toggle(
                "Enable Chaos Mode (Random Events)", 
                value=False, 
                disabled=True,
                help="This feature will introduce random market shocks and events in a future update."
    )
            st.caption("✨ Coming Soon!")
        st.divider()
        st.markdown("""
<style>
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #d9534f; /* Kırmızı bir ton */
    color: white; /* Yazı rengini beyaz yapıyoruz */
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #c9302c;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)
        if st.button("⚔️ Begin the Battle!", use_container_width=True, type="primary"):
            if not st.session_state.get("OPENAI_API_KEY"):
                st.error("Please enter your OpenAI API key in the sidebar to begin.")
            else:
                st.session_state.battle_config = {
                    "arena_settings": {"duration_weeks": num_weeks, "chaos_mode_enabled": chaos_mode},
                    "gladiators": st.session_state.gladiators
                }
                st.session_state.battle_started = True
                st.rerun()
    else:
        # If the battle is complete, show results
        if st.session_state.final_results_df is not None:
            render_battle_results(
                st.session_state.final_results_df, 
                st.session_state.final_war_chests,
                st.session_state.battle_config['gladiators']  # Gladyatör konfigürasyonunu buraya ekliyoruz
            )

            st.divider()
            st.subheader("Final State of the Arena")

            eliminated_final = set(
                g['name'] for g in st.session_state.battle_config['gladiators']
                if st.session_state.final_war_chests.get(g['name'], 0) < ELIMINATION_THRESHOLD
            )
            # Re-render final charts without live status cards or logs
            update_live_view(
                st.empty(),  # Empty placeholder for gladiator cards
                st.container(), # A real container for the charts
                st.empty(),  # Empty placeholder for combat log
                st.session_state.final_results_df,
                st.session_state.final_war_chests,
                eliminated_final,
                ["**Final Week:** The battle has concluded. See the leaderboard above."],
                st.session_state.battle_config['gladiators'],
                CHART_COLORS[:len(st.session_state.battle_config['gladiators'])],
                week_message=None
            )

            if st.button("🛡️ Start New Battle"):
                for key in ['gladiators', 'battle_started', 'final_results_df', 'battle_config', 'final_war_chests']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        # Otherwise, run the live battle
        else:
            run_live_battle(st.session_state.battle_config)
            # Once the battle is done, rerun to switch to the results view
            st.rerun()

    
if __name__ == "__main__":
    main()