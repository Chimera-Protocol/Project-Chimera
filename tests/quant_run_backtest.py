# =============================================================================
# tests/quant_run_backtest.py 
#
# Description:
#   This is the master long-term backtesting script for Project Chimera.
#   It is fully self-contained, using Alpaca for data and performing all
#   feature calculations internally. It runs the "Slow Strategy + ATR"
#   for 200 days and includes all final reporting enhancements.
# =============================================================================

# --- Section 1: Imports ---
import os
import sys
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import textwrap

# --- Section 2: Project-Specific Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import MarketSimulatorV2, SymbolicGuardianV6, CausalEngineV7_Quant
from src.config import FEATURE_COLS_DEFAULT

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# =============================================================================
# --- Section 3: Configuration & Initialization ---
# =============================================================================

print("--- Initializing Master Backtest (200 Days, Slow+ATR Strategy) ---")
load_dotenv()

# --- Backtest Configuration ---
SIMULATION_DAYS = 200
AGENT_MODEL = "gpt-4o"
INITIAL_CAPITAL = 100000.0

# --- API Configuration ---
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
if not API_KEY or not SECRET_KEY or not os.getenv("OPENAI_API_KEY"):
    raise ValueError("FATAL ERROR: Alpaca and OpenAI API keys must be set in your .env file.")

api = tradeapi.REST(API_KEY, SECRET_KEY, base_url="https://paper-api.alpaca.markets", api_version='v2')
print("✅ Successfully connected to Alpaca API for data fetching.")

# --- Chimera Component Initialization ---
guardian = SymbolicGuardianV6()
causal_engine = CausalEngineV7_Quant(data_path="causal_training_data_balanced.csv")
current_state_for_tools = {}
print("✅ Guardian and Causal Engine are ready.")


# =============================================================================
# --- Section 4: Data Preparation & Feature Engineering ---
# =============================================================================

def prepare_backtest_data(history_days: int = 500):
    """
    (BULLETPROOF VERSION)
    Fetches historical data from Alpaca and manually calculates all core features.
    """
    print(f"\n[Data Prep] Fetching last {history_days} days of data from Alpaca...")
    start_date = (datetime.now() - timedelta(days=history_days)).strftime('%Y-%m-%d')
    bars_df = api.get_crypto_bars('BTC/USD', '1Day', start=start_date).df
    
    if bars_df.empty or len(bars_df) < 100:
        raise ValueError(f"CRITICAL ERROR: Insufficient data from Alpaca.")
    
    print(f"[Data Prep] Calculating {len(FEATURE_COLS_DEFAULT)} core features (SLOW + ATR)...")
    bars_df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True, errors='ignore')

    # --- MANUAL FEATURE CALCULATION ---
    # (Existing calculations remain the same)
    delta = bars_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    bars_df['Rsi_14'] = 100 - (100 / (1 + rs))
    
    ema_fast = bars_df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = bars_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    bars_df['Macdh_12_26_9'] = macd_line - signal_line
    
    bars_df['Roc_15'] = bars_df['close'].pct_change(periods=15)
    
    sma20 = bars_df['close'].rolling(window=20).mean()
    sma100 = bars_df['close'].rolling(window=100).mean()
    bars_df['Price_vs_sma20'] = bars_df['close'] / sma20
    bars_df['Sma20_vs_sma100'] = sma20 / sma100
    
    std20 = bars_df['close'].rolling(window=20).std()
    bbu = sma20 + (std20 * 2)
    bbl = sma20 - (std20 * 2)
    bb_range = bbu - bbl
    bb_range[bb_range == 0] = np.nan
    bars_df['Bb_position'] = (bars_df['close'] - bbl) / bb_range
    
    # --- Manuel ATR Calculation ---
    print("[Data Prep] Calculating ATR for volatility analysis...")
    
    # First, calculate the three components of True Range
    high_low = bars_df['high'] - bars_df['low']
    high_prev_close = np.abs(bars_df['high'] - bars_df['close'].shift(1))
    low_prev_close = np.abs(bars_df['low'] - bars_df['close'].shift(1))
    
    # Combine them into a temporary DataFrame
    tr_df = pd.concat([high_low, high_prev_close, low_prev_close], axis=1)
    
    # True Range is the maximum of the three components for each day
    true_range = tr_df.max(axis=1)
    
    # Calculate the Average True Range (ATR) using a 14-period EMA (Exponential Moving Average)
    bars_df['Atr_14'] = true_range.ewm(alpha=1/14, adjust=False).mean()
    
    # Finalize column names to match components' expectations
    bars_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True, errors='ignore')
    bars_df.dropna(inplace=True)
    
    print("✅ SLOW STRATEGY + ATR data preparation complete.")
    return bars_df


# =============================================================================
# --- Section 5: Agent, Reporting, and Simulation Engine ---
# =============================================================================
# (Agent Tools and Reporting Functions are all included here)

# --- Agent Tools Definition ---
@tool
def check_action_validity(action_type: str, amount: float) -> str:
    """Checks if a proposed trading action is valid according to fundamental risk management rules."""
    action = {'type': action_type.upper(), 'amount': amount}
    state_for_guardian = {"cash": current_state_for_tools.get('cash'), "shares_held": current_state_for_tools.get('shares_held'), "market_data": current_state_for_tools.get('market_data')}
    return json.dumps(guardian.validate_action(action, state_for_guardian))

@tool
def estimate_profit_impact(action_type: str, amount: float) -> str:
    """Estimates the causal profit impact of a VALID trading action based on historical data patterns."""
    action = {'type': action_type.upper(), 'amount': amount}
    market_context = current_state_for_tools.get('market_data', {})
    if not market_context: return json.dumps({'error': 'Market context is empty.'})
    effect = causal_engine.estimate_causal_effect(action, market_context)
    return json.dumps({'predicted_profit_impact': effect})

# --- Daily Reporting Function ---
def print_daily_report(day, sim_state_before, sim_state_after, agent_response, action_data, total_days, initial_capital):
    # (The full, corrected code for the beautiful report)
    market_data = sim_state_after.get('market_data', {}); value_before = sim_state_before.get('portfolio_value', 0); value_after = sim_state_after.get('portfolio_value', 0); cash_after = sim_state_after.get('cash', 0); shares_after = sim_state_after.get('shares_held', 0); commentary = action_data.get('commentary', 'No commentary provided.'); action = action_data.get('action', {}); action_type = action.get('type', 'N/A').upper(); action_amount = action.get('amount', 0.0)
    daily_pnl = value_after - value_before; daily_pnl_pct = (daily_pnl / value_before) if value_before != 0 else 0.0; total_pnl = value_after - initial_capital; total_pnl_pct = (total_pnl / initial_capital) if initial_capital != 0 else 0.0
    HEADER = '\033[95m\033[1m'; CYAN = '\033[96m'; GREEN = '\033[92m'; RED = '\033[91m'; YELLOW = '\033[93m'; ENDC = '\033[0m'; BOLD = '\033[1m'
    pnl_color = GREEN if daily_pnl >= 0 else RED; total_pnl_color = GREEN if total_pnl >= 0 else RED; pnl_icon = "📈" if daily_pnl >= 0 else "📉"; total_pnl_icon = "🚀" if total_pnl >= 0 else "💥"; action_color = GREEN if action_type == 'BUY' else RED if action_type in ['SELL', 'SHORT'] else YELLOW; action_icon = "✅" if action_type in ['BUY', 'SHORT'] else "❌" if action_type == 'SELL' else "ℹ️"
    header_text = f"| {HEADER}🚀 CHIMERA DAILY REPORT | DAY {day+1}/{total_days}{ENDC}"
    print("\n" + "="*80); print(header_text.ljust(99) + "|"); print("="*80); print("|")
    price_str = f"${market_data.get('Close', 0):,.2f}".ljust(15); rsi_str = f"{market_data.get('Rsi_14', 0):.2f}".ljust(6); macd_str = f"{market_data.get('Macdh_12_26_9', 0):.2f}"
    print(f"|   {BOLD}📊 MARKET SNAPSHOT{ENDC}"); print(f"|   {'------------------'}"); print(f"|   BTC Price: {CYAN}{price_str}{ENDC}|  RSI: {CYAN}{rsi_str}{ENDC}|  MACD Hist: {CYAN}{macd_str}{ENDC}"); print("|")
    print(f"|   {BOLD}💡 AGENT DECISION{ENDC}"); print(f"|   {'-----------------'}"); wrapped_commentary = textwrap.fill(commentary, width=70).replace('\n', '\n|               '); print(f"|   Commentary: {wrapped_commentary}"); print(f"|   Action:     {action_color}{action_icon} {action_type} {action_amount:.2%} of available assets.{ENDC}"); print("|")
    value_change_str = f"${value_before:,.2f} -> ${value_after:,.2f}"; pnl_str = f"{pnl_color}{daily_pnl:+.2f} ({daily_pnl_pct:+.2%}) {pnl_icon}{ENDC}"; total_pnl_str = f"{total_pnl_color}{total_pnl:+.2f} ({total_pnl_pct:+.2%}) {total_pnl_icon}{ENDC}"
    print(f"|   {BOLD}💼 PORTFOLIO IMPACT{ENDC}"); print(f"|   {'--------------------'}"); print(f"|   Value Change:    {YELLOW}{value_change_str}{ENDC}"); print(f"|   Daily P&L:       {pnl_str}"); print(f"|   Total P&L:       {total_pnl_str}"); print("|"); print("|   Final Status:"); print(f"|   - Cash:          {YELLOW}${cash_after:,.2f}{ENDC}"); print(f"|   - Shares:        {YELLOW}{shares_after:.4f} BTC{ENDC}"); print("="*80)

# --- Final Charting Function ---
def analyze_and_report(history_df, actions_df, market_data_df, initial_capital):
    """
    Generates and saves a professional, multi-panel performance report dashboard.
    """
    print("\n--- Generating Final Performance Report ---")
    
    # --- Data Preparation for Plotting ---
    report_df = history_df.copy()
    report_df['day'] = range(len(report_df))
    report_df['btc_price'] = market_data_df['Close'].values
    report_df['daily_return'] = report_df['portfolio_value'].pct_change().fillna(0)
    
    # --- Calculate Advanced Metrics ---
    final_value = report_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    # Buy & Hold return
    buy_hold_return = (report_df['btc_price'].iloc[-1] / report_df['btc_price'].iloc[0]) - 1
    
    # Sharpe Ratio (annualized)
    risk_free_rate = 0.0
    mean_daily_return = report_df['daily_return'].mean()
    std_daily_return = report_df['daily_return'].std()
    sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return * np.sqrt(252) if std_daily_return != 0 else 0
    
    # Max Drawdown
    cumulative_roll_max = report_df['portfolio_value'].cummax()
    drawdown = report_df['portfolio_value'] / cumulative_roll_max - 1.0
    max_drawdown = drawdown.min()
    
    # Trade Statistics
    trades = actions_df[actions_df['type'] != 'HOLD']
    num_trades = len(trades)
    
    # --- Plotting Dashboard ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 28))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    fig.suptitle('Chimera-Quant: Final Backtest Performance Analysis', fontsize=28, weight='bold')

    # Panel 1: Cumulative Returns vs Buy & Hold
    ax1 = fig.add_subplot(gs[0, :])
    report_df['cumulative_strategy'] = (1 + report_df['daily_return']).cumprod()
    report_df['cumulative_bh'] = report_df['btc_price'] / report_df['btc_price'].iloc[0]
    ax1.plot(report_df['day'], report_df['cumulative_strategy'], label='Strategy Cumulative Return', color='darkgreen', lw=2.5)
    ax1.plot(report_df['day'], report_df['cumulative_bh'], label='Buy & Hold Cumulative Return', color='gray', linestyle='--', lw=2)
    ax1.set_title('1. Cumulative Returns: Strategy vs Buy & Hold', fontsize=18, weight='bold')
    ax1.set_ylabel('Cumulative Growth (x)', fontsize=14)
    ax1.set_xlabel('Trading Day', fontsize=14)
    ax1.legend(loc='upper left')

    # Panel 2: Strategy & Action Map on Price
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.plot(report_df['day'], report_df['btc_price'], color='lightgray', linestyle='-', alpha=0.9, lw=2, label='BTC Price')
    action_colors = {"BUY": "#2ca02c", "SELL": "#d62728", "SHORT": "#9467bd"}
    for _, action in trades.iterrows():
        marker = '^' if action['type'] == 'BUY' else 'v' if action['type'] == 'SELL' else 'o'
        color = action_colors[action['type']]
        ax2.scatter(action['day'], report_df.loc[action['day'], 'btc_price'], 
                    color=color, marker=marker, s=action['amount']*500 + 150, 
                    edgecolors='white', linewidth=1.5, zorder=10, label=action['type'])
    ax2.set_title('2. Strategy Map: Agent Decisions on Price Chart', fontsize=18, weight='bold')
    ax2.set_ylabel('BTC Price ($)', fontsize=14)
    ax2.set_xlabel('Trading Day', fontsize=14)
    
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper left', ncol=4)

    # Panel 3: Key Performance Indicators
    ax3 = fig.add_subplot(gs[2, 0])
    metrics_text = (
        f"----- Performance -----\n"
        f"Total Return: {total_return:.2%}\n"
        f"Buy & Hold Return: {buy_hold_return:.2%}\n\n"
        f"----- Risk -----\n"
        f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}\n"
        f"Max Drawdown: {max_drawdown:.2%}\n\n"
        f"----- Activity -----\n"
        f"Total Trades: {num_trades}\n"
    )
    ax3.text(0.5, 0.5, metrics_text, fontsize=18, va='center', ha='center', linespacing=1.8,
             bbox=dict(boxstyle="round,pad=0.5", fc="#D3D3D3", alpha=1))
    ax3.set_title('3. Key Performance Indicators', fontsize=18, weight='bold')
    ax3.axis('off')

    # Panel 4: Daily Returns Distribution
    ax4 = fig.add_subplot(gs[2, 1])
    sns.histplot(data=report_df, x='daily_return', ax=ax4, color='orchid', bins=30, kde=True)
    ax4.set_title('4. Daily Returns Distribution', fontsize=18, weight='bold')
    ax4.set_xlabel('Daily Return (%)', fontsize=14)
    ax4.axvline(0, color='white', linestyle='--', lw=1)

    # Panel 5: Portfolio Growth vs Market
    ax5 = fig.add_subplot(gs[3, :])
    ax5.plot(report_df['day'], report_df['portfolio_value'], color='darkgreen', label='Chimera Portfolio Value', lw=2.5, zorder=5)
    ax5.set_ylabel('Portfolio Value ($)', color='darkgreen', fontsize=14)
    ax5.set_title('5. Portfolio Growth vs. Market', fontsize=18, weight='bold')
    ax5.tick_params(axis='y', labelcolor='darkgreen')
    ax5.axhline(initial_capital, color='white', linestyle='--', lw=1, alpha=0.7)
    
    ax5_twin = ax5.twinx()
    ax5_twin.plot(report_df['day'], report_df['btc_price'], color='gray', linestyle=':', label='BTC Price', lw=1.5, alpha=0.8)
    ax5_twin.set_ylabel('BTC Price ($)', color='gray', fontsize=14)
    ax5_twin.tick_params(axis='y', labelcolor='gray')
    
    # Combine legends
    lines, labels = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    # Save the final report
    os.makedirs("results/quant", exist_ok=True)
    output_path = os.path.join("results/quant", "final_backtest_report.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Professional performance report saved to '{output_path}'")
    plt.show()



# =============================================================================
# --- Section 6: Main Execution Block ---
# =============================================================================

if __name__ == '__main__':
    full_market_data = prepare_backtest_data()
    simulation_data = full_market_data.tail(SIMULATION_DAYS).reset_index(drop=True)

    SYSTEM_PROMPT = """You are "Chimera-Quant", a world-class autonomous trading agent.
    Your decision-making process is a strict, non-negotiable workflow.

    **STRATEGIC RULES:**
    **1. Core Analysis:** Your primary goal is to analyze market trends using RSI, MACD, SMAs, etc.
    **2. Volatility Awareness (ATR):** `Atr_14` measures market volatility. A sudden, massive spike in ATR during a sell-off can signal panic and a potential trend reversal (a "V-shaped" recovery). If your other indicators suggest a SHORT, but ATR is extremely high or spiking, **BE CAUTIOUS**. A smaller position size or waiting for confirmation might be a wiser move than an aggressive SHORT.

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
    
    tools = [check_action_validity, estimate_profit_impact]
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "Current State:\n{state_json}"), MessagesPlaceholder(variable_name="agent_scratchpad")])
    llm = ChatOpenAI(model=AGENT_MODEL, temperature=0.1)
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors="raise")
    
    print(f"\n--- Starting {SIMULATION_DAYS}-Day Backtest ---")
    simulator = MarketSimulatorV2(market_data=simulation_data, initial_capital=INITIAL_CAPITAL)
    history, actions_log = [], []

    for day in tqdm(range(len(simulation_data)), desc="Simulating Trading Days"):
        sim_state_before = simulator.get_state()
        history.append(sim_state_before)
        current_state_for_tools = sim_state_before
        
        response = {}; decision = {}
        try:
            response = executor.invoke({"state_json": json.dumps(sim_state_before, default=str)})
            final_json_str = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response['output'])
            decision = json.loads(final_json_str.group(1)) if final_json_str else {}
            action = decision.get('action', {})
            final_action_type = action.get('type', 'HOLD')
            final_amount = float(action.get('amount', 0.0))
        except Exception as e:
            print(f"\nError on day {day+1}: {e}. Defaulting to HOLD.")
            final_action_type, final_amount = 'HOLD', 0.0
            decision = {"commentary": f"An error occurred: {e}", "action": {"type": "HOLD", "amount": 0.0}}
        
        simulator.step({'type': final_action_type, 'amount': final_amount})
        actions_log.append({'day': day, 'type': final_action_type, 'amount': final_amount})
        
        sim_state_after = simulator.get_state()
        print_daily_report(day, sim_state_before, sim_state_after, response, decision, SIMULATION_DAYS, INITIAL_CAPITAL)

    print("\n--- Backtest Complete ---")
    history_df = pd.DataFrame(history)
    actions_df = pd.DataFrame(actions_log)
    analyze_and_report(history_df, actions_df, simulation_data, INITIAL_CAPITAL)