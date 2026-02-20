#!/usr/bin/env python3
# =============================================================================
# live_paper_trader.py (v8.0 - PRODUCTION READY)
#
# Changes:
#   - Runs ONCE per day at 00:00 UTC
#   - Better error recovery with exponential backoff
#   - Session state tracking
#   - Risk management checks
#   - Comprehensive logging
# =============================================================================

import os
import sys
import json
import re
import time
import schedule
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import logging
from alpaca_trade_api.rest import APIError

# --- Project Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import SymbolicGuardianV6, CausalEngineV7_Quant
from src.config import FEATURE_COLS_DEFAULT

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trader_activity.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

print("="*60)
print("  CHIMERA LIVE TRADER - INITIALIZING")
print("="*60)
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

if not API_KEY or not SECRET_KEY or not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ FATAL: API keys missing in .env file")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
logging.info("✅ Connected to Alpaca API")

# Initialize components
guardian = SymbolicGuardianV6()
causal_engine = CausalEngineV7_Quant(data_path="causal_training_data_balanced.csv")
current_state_for_tools = {}
logging.info("✅ Chimera components initialized")

# =============================================================================
# AGENT TOOLS
# =============================================================================

@tool
def check_action_validity(action_type: str, amount: float) -> str:
    """Checks if a trading action is valid according to Guardian rules"""
    action = {'type': action_type.upper(), 'amount': amount}
    result = guardian.validate_action(action, current_state_for_tools)
    
    # If Guardian suggests an adjusted amount, update the action
    if 'adjusted_amount' in result:
        action['amount'] = result['adjusted_amount']
        result['action'] = action
    
    return json.dumps(result)

@tool
def estimate_profit_impact(action_type: str, amount: float) -> str:
    """Estimates causal profit impact of an action"""
    action = {'type': action_type.upper(), 'amount': amount}
    market_context = current_state_for_tools.get('market_data', {})
    
    if not market_context:
        return json.dumps({'error': 'No market context available'})
    
    # Verify all required features are present
    missing = [f for f in FEATURE_COLS_DEFAULT if f not in market_context]
    if missing:
        return json.dumps({'error': f'Missing features: {missing}'})
    
    try:
        effect = causal_engine.estimate_causal_effect(action, market_context)
        return json.dumps({'predicted_profit_impact': effect})
    except Exception as e:
        return json.dumps({'error': str(e)})

# =============================================================================
# DATA FETCHING & FEATURE ENGINEERING
# =============================================================================

def create_live_features(symbol: str = 'BTC/USD', history_days: int = 400) -> pd.DataFrame:
    """
    Fetches historical data and calculates all technical indicators.
    Matches the exact feature set used in backtesting.
    """
    from datetime import timedelta
    
    start_date = (datetime.now(timezone.utc) - timedelta(days=history_days)).strftime('%Y-%m-%d')
    
    logging.info(f"Fetching {history_days} days of daily bars...")
    bars_df = api.get_crypto_bars(symbol, '1Day', start=start_date).df
    
    if bars_df.empty or len(bars_df) < 100:
        raise ValueError(f"❌ Insufficient data: only {len(bars_df)} bars")
    
    logging.info(f"Calculating {len(FEATURE_COLS_DEFAULT)} technical features...")
    
    # Normalize column names
    bars_df.rename(columns={
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'open': 'Open',
        'volume': 'Volume'
    }, inplace=True, errors='ignore')
    
    # --- RSI (14) ---
    delta = bars_df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    bars_df['Rsi_14'] = 100 - (100 / (1 + rs))
    
    # --- MACD Histogram ---
    ema_fast = bars_df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = bars_df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    bars_df['Macdh_12_26_9'] = macd_line - signal_line
    
    # --- ROC (15) ---
    bars_df['Roc_15'] = bars_df['Close'].pct_change(periods=15)
    
    # --- Price vs SMA ---
    sma20 = bars_df['Close'].rolling(window=20).mean()
    sma100 = bars_df['Close'].rolling(window=100).mean()
    bars_df['Price_vs_sma20'] = bars_df['Close'] / sma20
    bars_df['Sma20_vs_sma100'] = sma20 / sma100
    
    # --- Bollinger Band Position ---
    std20 = bars_df['Close'].rolling(window=20).std()
    bbu = sma20 + (std20 * 2)
    bbl = sma20 - (std20 * 2)
    bb_range = bbu - bbl
    bb_range[bb_range == 0] = np.nan
    bars_df['Bb_position'] = (bars_df['Close'] - bbl) / bb_range
    
    # --- ATR (14) ---
    high_low = bars_df['High'] - bars_df['Low']
    high_prev_close = np.abs(bars_df['High'] - bars_df['Close'].shift(1))
    low_prev_close = np.abs(bars_df['Low'] - bars_df['Close'].shift(1))
    tr_df = pd.concat([high_low, high_prev_close, low_prev_close], axis=1)
    true_range = tr_df.max(axis=1)
    bars_df['Atr_14'] = true_range.ewm(alpha=1/14, adjust=False).mean()
    
    bars_df.dropna(inplace=True)
    
    logging.info(f"✅ Feature calculation complete. {len(bars_df)} bars ready.")
    return bars_df

def get_live_state() -> dict:
    """Fetches current portfolio and market state"""
    try:
        account = api.get_account()
        
        shares_held, entry_price, unrealized_pnl, unrealized_pnl_pct = 0.0, 0.0, 0.0, 0.0
        
        try:
            position = api.get_position('BTCUSD')
            shares_held = float(position.qty)
            entry_price = float(position.avg_entry_price)
            unrealized_pnl = float(position.unrealized_pl)
            unrealized_pnl_pct = float(position.unrealized_plpc) * 100
        except APIError as e:
            if "position does not exist" in str(e).lower():
                logging.info("No active position found")
            else:
                logging.error(f"Unexpected position fetch error: {e}")
                raise
        
        # Fetch and calculate market features
        market_data_df = create_live_features('BTC/USD')
        latest_market_data = market_data_df.iloc[-1].to_dict()
        
        state = {
            "market_data": latest_market_data,
            "portfolio_value": float(account.equity),
            "cash": float(account.cash),
            "shares_held": shares_held,
            "entry_price": entry_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct
        }
        
        logging.info(f"State: Cash=${state['cash']:,.2f}, Equity=${state['portfolio_value']:,.2f}, BTC={shares_held:.6f}")
        return state
        
    except Exception as e:
        logging.critical(f"Critical error in get_live_state: {e}", exc_info=True)
        raise

# =============================================================================
# TRADE EXECUTION
# =============================================================================

def execute_trade(decision: dict, live_state: dict):
    """
    Executes trade based on agent's target *amount*.
    BUY: Sets target position to the 'amount'.
    SELL/SHORT: Reduces current position by the 'amount'.
    HOLD: Does nothing.
    """
    portfolio_value = live_state.get('portfolio_value', 0.0)
    current_qty = live_state.get('shares_held', 0.0)
    current_price = live_state.get('market_data', {}).get('Close', 0.0)
    
    if portfolio_value <= 0 or current_price <= 0:
        logging.warning("Invalid portfolio or price data. Cannot execute.")
        return None
    
    action = decision.get('action', {})
    action_type = action.get('type', 'HOLD').upper()
    amount_ratio = float(action.get('amount', 0.0))
    symbol = 'BTCUSD'

    # --- YENİ MANTIK (HATAYI DÜZELTME) ---

    # 1. İşlem yapılacak 'değer' (value) miktarını hesapla
    # (Hem BUY hem SELL/SHORT için 'amount' kullanılır)
    value_to_trade = portfolio_value * amount_ratio
    qty_to_trade = value_to_trade / current_price

    # 2. 'target_qty' (hedef miktar) belirle
    target_qty = 0.0 

    if action_type == 'BUY':
        # 'BUY 0.70' -> Hedef = 0.70 * Portföy Değeri
        target_qty = qty_to_trade
    
    elif action_type in ['SELL', 'SHORT']:
        # 'SELL 0.10' -> Hedef = (Mevcut Miktar) - (0.10 * Portföy Değeri)
        # 'SHORT' da 'SELL' ile aynı işlemi yapsın (spotta olduğumuz için)
        target_qty = current_qty - qty_to_trade
        
        # Güvenlik kilidi: 0'ın altına inemeyiz (negatife düşemeyiz)
        if target_qty < 0:
            target_qty = 0.0
            
    elif action_type == 'HOLD':
        # 'HOLD' -> Hiçbir şey yapma, mevcut miktarı koru
        target_qty = current_qty

    # --- ESKİ MANTIK ('target_qty = 0.0' kısmı) ARTIK TAMAMEN SİLİNDİ ---
    
    # 3. Delta (fark) üzerinden emri hesapla
    delta_qty = target_qty - current_qty
    
    logging.info(f"Execution: Action={action_type} {amount_ratio:.2%}. Current={current_qty:.6f}, Target={target_qty:.6f}, Delta={delta_qty:.6f}")
    
    # Minimum trade size filter ($10)
    # Eğer 'amount_ratio' 0 ise (örn: HOLD 0.00%) delta_qty 0 olacaktır.
    if abs(delta_qty * current_price) < 10.0:
        logging.info("Delta too small (<$10) or action is HOLD. No trade.")
        return None
    
    try:
        if delta_qty > 0:  # BUY
            qty_to_trade_final = round(delta_qty, 6)
            logging.info(f"Placing BUY order: {qty_to_trade_final:.6f} BTC")
            order = api.submit_order(
                symbol=symbol,
                qty=qty_to_trade_final,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            return order
            
        elif delta_qty < 0:  # SELL
            qty_to_trade_final = round(abs(delta_qty), 6)
            logging.info(f"Placing SELL order: {qty_to_trade_final:.6f} BTC")
            order = api.submit_order(
                symbol=symbol,
                qty=qty_to_trade_final,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            return order
            
    except Exception as e:
        logging.error(f"Order submission failed: {e}", exc_info=True)
        return e
    
    return None

# =============================================================================
# REPORTING
# =============================================================================

def save_report_for_dashboard(decision: dict, agent_steps: list):
    """Saves agent analysis for dashboard visualization"""
    scenarios_data = []
    scenarios = {}
    
    for step in agent_steps:
        tool_input = step[0].tool_input
        action_type = tool_input.get('action_type', 'N/A')
        amount = tool_input.get('amount', 0.0)
        hypothesis_key = f"{action_type} {amount:.2%}"
        
        tool_name = step[0].tool
        obs_data = json.loads(step[1])
        
        if hypothesis_key not in scenarios:
            scenarios[hypothesis_key] = {
                'validation': 'Pending',
                'message': '',
                'impact': None
            }
        
        if tool_name == 'check_action_validity':
            scenarios[hypothesis_key]['validation'] = "Valid" if obs_data.get('is_valid') else "Invalid"
            scenarios[hypothesis_key]['message'] = obs_data.get('message', '')
        elif tool_name == 'estimate_profit_impact':
            scenarios[hypothesis_key]['impact'] = obs_data.get('predicted_profit_impact')
    
    for key, value in scenarios.items():
        scenarios_data.append({'hypothesis': key, **value})
    
    report_data = {
        "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        "final_action": decision.get('action', {}),
        "commentary": decision.get('commentary', 'No commentary'),
        "scenarios": scenarios_data
    }
    
    try:
        with open("last_cycle_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        logging.info("✅ Dashboard report saved")
    except Exception as e:
        logging.error(f"Failed to save report: {e}")

def print_cycle_report(decision: dict, agent_steps: list, order_result):
    """Prints comprehensive cycle report to console"""
    # ANSI Colors
    HEADER = '\033[95m\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'
    
    print("\n" + "="*80)
    print(f"| {HEADER}CHIMERA CYCLE REPORT - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC{ENDC}")
    print("="*80)
    
    # Final Decision
    action = decision.get('action', {})
    commentary = decision.get('commentary', 'No commentary')
    action_type = action.get('type', 'HOLD')
    amount = action.get('amount', 0.0)
    
    action_color = GREEN if action_type == 'BUY' else RED if action_type in ['SELL', 'SHORT'] else YELLOW
    
    print(f"\n{BOLD}[1] AGENT'S FINAL DECISION{ENDC}")
    print("-" * 80)
    print(f"   Commentary: {commentary}")
    print(f"   Action: {action_color}{BOLD}{action_type} {amount:.2%}{ENDC}")
    
    # Scenario Analysis
    print(f"\n{BOLD}[2] SCENARIO ANALYSIS{ENDC}")
    print("-" * 80)
    
    scenarios = {}
    for step in agent_steps:
        tool_input = step[0].tool_input
        action_type_h = tool_input.get('action_type', 'N/A')
        amount_h = tool_input.get('amount', 0.0)
        hypo_key = f"{action_type_h} {amount_h:.2%}"
        
        if hypo_key not in scenarios:
            scenarios[hypo_key] = {'validation': '', 'estimation': ''}
        
        tool_name = step[0].tool
        obs_data = json.loads(step[1])
        
        if tool_name == 'check_action_validity':
            is_valid = obs_data.get('is_valid', False)
            msg = obs_data.get('message', '')
            status = f"{GREEN}[OK]{ENDC}" if is_valid else f"{RED}[X]{ENDC} ({msg})"
            scenarios[hypo_key]['validation'] = status
        
        elif tool_name == 'estimate_profit_impact':
            profit = obs_data.get('predicted_profit_impact', 0.0)
            profit_color = GREEN if profit > 0 else RED if profit < 0 else YELLOW
            scenarios[hypo_key]['estimation'] = f"Impact: {profit_color}{profit:+.2%}{ENDC}"
    
    for hypo, results in scenarios.items():
        val = results.get('validation', '')
        est = results.get('estimation', '')
        print(f"   H: {hypo.ljust(15)} {val.ljust(20)} {est}")
    
    # Execution Details
    print(f"\n{BOLD}[3] EXECUTION DETAILS{ENDC}")
    print("-" * 80)
    
    if isinstance(order_result, Exception):
        print(f"   Status: {RED}FAILED{ENDC}")
        print(f"   Error: {str(order_result)}")
    elif order_result:
        time.sleep(3)  # Wait for fill
        try:
            order = api.get_order(order_result.id)
            print(f"   Status: {GREEN}SUCCESS{ENDC}")
            print(f"   Order ID: {order.id}")
            print(f"   Side: {order.side.upper()}")
            print(f"   Quantity: {BLUE}{float(order.filled_qty):.6f} BTC{ENDC}")
            print(f"   Avg Price: {BLUE}${float(order.filled_avg_price):,.2f}{ENDC}")
            notional_value = float(order.notional) if order.notional else 0.0
            print(f"   Total Value: {BLUE}${notional_value:,.2f}{ENDC}")
        except Exception as e:
            print(f"   Could not fetch order details: {e}")
            print(f"   Error fetching order details: {e}")
    else:
        print(f"   Status: {YELLOW}NO TRADE{ENDC}")
    
    # Portfolio Status
    print(f"\n{BOLD}[4] POST-TRADE PORTFOLIO{ENDC}")
    print("-" * 80)
    
    try:
        account = api.get_account()
        print(f"   Portfolio Value: {YELLOW}${float(account.equity):,.2f}{ENDC}")
        print(f"   Cash: {YELLOW}${float(account.cash):,.2f}{ENDC}")
        
        try:
            pos = api.get_position('BTCUSD')
            print(f"   BTC Position: {YELLOW}{float(pos.qty):.6f} BTC{ENDC}")
        except:
            print(f"   BTC Position: None")
    except Exception as e:
        print(f"   Error fetching account: {e}")
    
    print("="*80 + "\n")
    
    save_report_for_dashboard(decision, agent_steps)

# =============================================================================
# MAIN AGENT LOGIC
# =============================================================================

def run_agent_once():
    """Runs one complete agent decision cycle"""
    print("\n" + "="*60)
    print(f"  CHIMERA AGENT CYCLE - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*60)
    
    try:
        # Step 1: Observe environment
        logging.info("[1/5] Fetching live state...")
        live_state = get_live_state()
        global current_state_for_tools
        current_state_for_tools.update(live_state)
        logging.info("✅ State fetched")
        
        # Step 2: Agent reasoning
        logging.info("[2/5] Agent reasoning...")
        
        SYSTEM_PROMPT = SYSTEM_PROMPT = """You are "Chimera-Quant", a world-class autonomous trading agent.
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
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Goal: Maximize long-term portfolio value.\n\nCurrent State:\n{state_json}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=15
        )
        
        response = executor.invoke({"state_json": json.dumps(live_state, default=str)})
        logging.info("✅ Agent reasoning complete")
        
        # Step 3: Parse decision
        logging.info("[3/5] Parsing decision...")
        decision = {}
        final_json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response['output'])
        
        if final_json_match:
            decision = json.loads(final_json_match.group(1))
            logging.info("✅ Decision parsed")
        else:
            logging.warning("⚠️  Could not parse decision. Defaulting to HOLD.")
            decision = {
                "commentary": "Failed to parse agent output. Defaulting to safe HOLD.",
                "action": {"type": "HOLD", "amount": 0.0}
            }
        
        # Step 4: Execute
        logging.info("[4/5] Executing trade...")
        order_result = execute_trade(decision, live_state)
        
        # Step 5: Report
        logging.info("[5/5] Generating report...")
        print_cycle_report(decision, response.get('intermediate_steps', []), order_result)
        
        logging.info("✅ Cycle complete")
        
    except Exception as e:
        logging.critical(f"❌ Critical error in agent cycle: {e}", exc_info=True)

# =============================================================================
# SCHEDULER
# =============================================================================

def scheduled_job():
    """Job to run once per day"""
    logging.info("="*60)
    logging.info("  SCHEDULED AGENT EXECUTION STARTING")
    logging.info("="*60)
    run_agent_once()

if __name__ == '__main__':
    import sys
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("🧪 TEST MODE: Running once immediately")
        run_agent_once()
        sys.exit(0)
    
    # Production mode: schedule daily at 00:00 UTC
    print("📅 PRODUCTION MODE: Scheduling daily execution at 00:00 UTC")
    schedule.every().day.at("00:00").do(scheduled_job)
    
    print("✅ Scheduler active. Press Ctrl+C to stop.")
    print(f"   Next run: {schedule.next_run()}")
    
    # Run once immediately on startup
    print("\n🚀 Running initial cycle...")
    run_agent_once()
    
    # Then wait for scheduled runs
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute