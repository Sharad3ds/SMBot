import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from fyers_apiv3 import fyersModel
from fyers_apiv3 import fyersModel as FyersV3
import logging
import os
from typing import Dict, Optional
import warnings
import pytz
import pandas_ta as ta
import calendar

warnings.filterwarnings("ignore", category=FutureWarning)
IST = pytz.timezone("Asia/Kolkata")


# --------------------------- EXPIRY HELPERS ---------------------------

def get_last_thursday(year: int, month: int) -> datetime:
    last_day = calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last_day)
    while dt.weekday() != 3:  # Thursday = 3
        dt -= timedelta(days=1)
    return dt


def format_expiry_code(dt: datetime) -> str:
    yy = dt.year % 100
    mon = dt.strftime("%b").upper()
    return f"{yy:02d}{mon}"


def get_current_or_next_monthly_expiry(now_ist: Optional[datetime] = None) -> datetime:
    """
    Use last Thursday of current month if it is today or in future,
    else switch to next month monthly expiry.
    """
    now_ist = now_ist or datetime.now(IST)
    y, m = now_ist.year, now_ist.month
    exp_dt = get_last_thursday(y, m)

    if exp_dt.date() < now_ist.date():
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
        exp_dt = get_last_thursday(y, m)
    return exp_dt


# --------------------------- CONFIGURATION ---------------------------

class TradingConfig:
    """Centralized configuration management"""

    UNDERLYING_SYMBOL = "NSE:NIFTY50-INDEX"
    TIMEFRAME = "1"          # 1‑min trading timeframe,   ll5
    
    
    
    
    
    
    
    
    4'//4,km ju  v

    # HA Doji strategy params                      b"'
    DOJI_MAX_BODY_PCT = 0.9
    OPEN_TOLERANCE    = 0.1

    LIMIT_BUFFER = 0.5

    # EMA trend filter
    USE_EMA_FILTER     = False
    EMA_LEN            = 21
    USE_EMA_SLOPE      = False
    EMA_SLOPE_BARS     = 5
    USE_SLOPE_STRENGTH = False
    SLOPE_LOOKBACK     = 10
    MIN_SLOPE_PERCENT  = 0.3

    # RSI filter
    USE_RSI_FILTER = False
    RSI_LEN        = 14
    RSI_OVERSOLD   = 30
    RSI_OVERBOUGHT = 70

    STRIKE_STEP    = 100
    # Updated to new NSE NIFTY lot size (Jan 2026): 65. [web:1][web:26]
    NIFTY_LOT_SIZE = 65

    # EMA9 angle strategy
    EMA9_LEN            = 21
    EMA9_ANGLE_LOOKBACK = 3
    EMA9_MIN_ANGLE_DEG  = 40.0

    # Higher timeframe (HTF) filter
    HTF_RULE    = "60T"
    HTF_EMA_LEN = 21

    # Trading session filter (NSE)
    SESSION_START = time(9, 20)
    SESSION_END   = time(15, 30)

    def __init__(self):
        now = datetime.now(IST)
        exp_dt = get_current_or_next_monthly_expiry(now)
        self.EXPIRY_CODE = format_expiry_code(exp_dt)
        logging.getLogger(__name__).info(
            f"Using monthly expiry {exp_dt.date()} with code {self.EXPIRY_CODE}"
        )


# --------------------------- LOGGING ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------- TRADING BOT ---------------------------

class TradingBot:
    def __init__(self):
        self.config = TradingConfig()
        self.fyers = self._init_fyers()
        self.position_size: int = 0
        self.last_trade: Optional[str] = None

    # INIT FYERS (v3 only, token pre-generated)
    def _init_fyers(self):
        if not all(os.path.exists(f) for f in ["fyers_appid.txt", "fyers_token.txt"]):
            raise FileNotFoundError("Missing fyers_appid.txt or fyers_token.txt")

        app_id = open("fyers_appid.txt", "r").read().strip()
        token = open("fyers_token.txt", "r").read().strip()

        if not app_id or not token:
            raise ValueError("Empty app_id or token in credential files")

        fyers_v3 = FyersV3.FyersModel(
            client_id=app_id,
            token=token,
            log_path=""
        )
        ver = getattr(fyers_v3, "version", "v3")
        logger.info(f"✅ Fyers V3 API initialized successfully (version={ver})")
        return fyers_v3

    # ---------------------- HISTORY HELPERS ----------------------

    def _fetch_history(self, symbol: str, days: int) -> pd.DataFrame:
        end_dt_ist = datetime.now(IST)
        start_dt_ist = end_dt_ist - timedelta(days=days)

        range_from = start_dt_ist.strftime("%Y-%m-%d")
        range_to   = end_dt_ist.strftime("%Y-%m-%d")

        data = {
            "symbol": symbol,
            "resolution": self.config.TIMEFRAME,
            "date_format": "1",
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1",
        }
        resp = self.fyers.history(data=data)

        if isinstance(resp, dict):
            if resp.get("s") != "ok":
                raise ValueError(f"API error for {symbol}: {resp}")
            candles = resp.get("candles", [])
        else:
            candles = resp.get("candles", []) if hasattr(resp, "get") else []

        if not candles:
            raise ValueError(f"No historical data for {symbol}")

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_index_data(self, days: int = None) -> pd.DataFrame:
        days = days or self.config.HIST_DAYS
        df = self._fetch_history(self.config.UNDERLYING_SYMBOL, days)
        df.rename(
            columns={
                "open": "und_open",
                "high": "und_high",
                "low": "und_low",
                "close": "und_close",
                "volume": "und_volume",
            },
            inplace=True,
        )
        df.set_index("timestamp", inplace=True)

        session_mask = df.index.indexer_between_time(
            self.config.SESSION_START, self.config.SESSION_END,
            include_start=True, include_end=True
        )
        df = df.iloc[session_mask]

        logger.info(f"✅ Loaded NIFTY50 index data with {len(df)} rows [IST] in session")
        return df

    def _fetch_option_history(self, symbol: str, days: int) -> pd.DataFrame:
        df = self._fetch_history(symbol, days)
        df.set_index("timestamp", inplace=True)
        return df

    # ---------------------- ITM OPTION PICKER ----------------------

    def _pick_itm_option_symbol(self, underlying_price: float, direction: str) -> str:
        """
        - BUY (bullish index)  -> ITM CE (strike < spot)
        - SELL (bearish index)-> ITM PE (strike > spot)
        """
        c = self.config
        spot = float(underlying_price)
        step = c.STRIKE_STEP

        if direction == "BUY":
            atm_strike = (int(spot) // step) * step
            strike = atm_strike - step
            opt_type = "CE"
        else:
            atm_strike = ((int(spot) + step - 1) // step) * step
            strike = atm_strike + step
            opt_type = "PE"

        strike = max(step, (int(strike) // step) * step)
        strike_int = int(strike)

        symbol = f"NSE:NIFTY{c.EXPIRY_CODE}{strike_int}{opt_type}"
        return symbol

    # ---------------------- INDICATORS ----------------------

    def calculate_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        base = df[["und_open", "und_high", "und_low", "und_close"]].rename(
            columns={
                "und_open": "open",
                "und_high": "high",
                "und_low": "low",
                "und_close": "close",
            }
        )
        ha = ta.ha(
            open_=base["open"],
            high=base["high"],
            low=base["low"],
            close=base["close"],
        )
        df_ha = df.copy()
        df_ha["ha_open"]  = ha["HA_open"]
        df_ha["ha_high"]  = ha["HA_high"]
        df_ha["ha_low"]   = ha["HA_low"]
        df_ha["ha_close"] = ha["HA_close"]
        return df_ha

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        df_ha = self.calculate_heikin_ashi(df)

        df_ha["ema"] = df_ha["und_close"].ewm(span=c.EMA_LEN, adjust=False).mean()

        bars = c.EMA_SLOPE_BARS
        df_ha["ema_slope_up"]   = df_ha["ema"] > df_ha["ema"].shift(bars)
        df_ha["ema_slope_down"] = df_ha["ema"] < df_ha["ema"].shift(bars)

        lookback = c.SLOPE_LOOKBACK
        ema_prev = df_ha["ema"].shift(lookback)
        df_ha["ema_change_pct"] = (df_ha["ema"] - ema_prev).abs() / ema_prev.abs() * 100
        df_ha["strong_trend"]   = df_ha["ema_change_pct"] >= c.MIN_SLOPE_PERCENT

        delta = df_ha["und_close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(c.RSI_LEN, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(c.RSI_LEN, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df_ha["rsi"] = 100 - (100 / (1 + rs))

        df_ha["ema9"] = df_ha["und_close"].ewm(span=c.EMA9_LEN, adjust=False).mean()
        n = c.EMA9_ANGLE_LOOKBACK
        ema9_prev = df_ha["ema9"].shift(n)
        slope = (df_ha["ema9"] - ema9_prev) / n
        slope_norm = slope / df_ha["ema9"].replace(0, np.nan)
        df_ha["ema9_angle_deg"] = np.degrees(np.arctan(slope_norm))

        return df_ha

    # ---------------------- HTF (60-min) TREND ----------------------

    def build_htf(self, df_idx: pd.DataFrame) -> pd.DataFrame:
        c = self.config

        df_15 = df_idx.resample(c.HTF_RULE).agg({
            "und_open": "first",
            "und_high": "max",
            "und_low": "min",
            "und_close": "last",
            "und_volume": "sum",
        }).dropna()

        df_15["htf_ema"] = df_15["und_close"].ewm(span=c.HTF_EMA_LEN, adjust=False).mean()
        df_15["htf_trend"] = np.where(
            df_15["und_close"] > df_15["htf_ema"], "up",
            np.where(df_15["und_close"] < df_15["htf_ema"], "down", "flat")
        )

        df_15 = df_15.reset_index()
        return df_15

    # ---------------------- REVERSAL ON ACTUAL CANDLES ----------------------

    def _is_bullish_reversal_actual(self, df: pd.DataFrame, i: int) -> bool:
        cur = df.iloc[i]
        prev = df.iloc[i - 1]

        cond_dir = (prev["und_close"] < prev["und_open"]) and (cur["und_close"] > cur["und_open"])

        rng = cur["und_high"] - cur["und_low"]
        if rng <= 0:
            return False

        body = abs(cur["und_close"] - cur["und_open"])
        upper_wick = cur["und_high"] - max(cur["und_close"], cur["und_open"])

        cond_structure = (
            cur["und_low"] <= prev["und_low"] and
            upper_wick <= 0.5 * rng and
            body       >= 0.2 * rng
        )
        return cond_dir and cond_structure

    def _is_bearish_reversal_actual(self, df: pd.DataFrame, i: int) -> bool:
        cur = df.iloc[i]
        prev = df.iloc[i - 1]

        cond_dir = (prev["und_close"] > prev["und_open"]) and (cur["und_close"] < cur["und_open"])

        rng = cur["und_high"] - cur["und_low"]
        if rng <= 0:
            return False

        body = abs(cur["und_close"] - cur["und_open"])
        lower_wick = min(cur["und_close"], cur["und_open"]) - cur["und_low"]

        cond_structure = (
            cur["und_high"] >= prev["und_high"] and
            lower_wick <= 0.5 * rng and
            body       >= 0.2 * rng
        )
        return cond_dir and cond_structure

    # ---------------------- SIGNAL GENERATION ----------------------

    def generate_signals(self, df_index_1: pd.DataFrame) -> pd.DataFrame:
        df_1 = df_index_1.copy().reset_index()

        df_ha = self.calculate_indicators(df_1)
        df_15 = self.build_htf(df_index_1)

        df_ha = pd.merge_asof(
            df_ha.sort_values("timestamp"),
            df_15[["timestamp", "htf_trend"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        signals = []

        warmup = max(
            self.config.EMA_LEN,
            self.config.SLOPE_LOOKBACK + 1,
            self.config.RSI_LEN + 1,
            self.config.EMA9_ANGLE_LOOKBACK + 1,
            2,
        )

        for i in range(warmup, len(df_ha)):
            signal = self._check_signal(df_ha, i)
            if signal:
                signals.append(signal)

        signals_df = pd.DataFrame(signals) if signals else pd.DataFrame()
        logger.info(f"📊 Generated {len(signals_df)} index-based signals (IST)")
        return signals_df

    def _check_signal(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        c = self.config
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        prev_body  = abs(prev["ha_close"] - prev["ha_open"])
        prev_range = prev["ha_high"] - prev["ha_low"]
        prev_is_doji = prev_range > 0 and (prev_body / prev_range) < c.DOJI_MAX_BODY_PCT

        fuzzy_match = abs(row["ha_open"] - prev["ha_close"]) / prev["ha_close"] < c.OPEN_TOLERANCE

        raw_long  = prev_is_doji and fuzzy_match and (row["ha_close"] > prev["ha_high"])
        raw_short = prev_is_doji and fuzzy_match and (row["ha_close"] < prev["ha_low"])

        ema = row["ema"]
        close = row["und_close"]
        ema_slope_up   = bool(row["ema_slope_up"])
        ema_slope_down = bool(row["ema_slope_down"])
        strong_trend   = bool(row["strong_trend"])

        ema_ok_long = (
            (not c.USE_EMA_FILTER or close > ema) and
            (not c.USE_EMA_SLOPE or ema_slope_up) and
            (not c.USE_SLOPE_STRENGTH or strong_trend)
        )
        ema_ok_short = (
            (not c.USE_EMA_FILTER or close < ema) and
            (not c.USE_EMA_SLOPE or ema_slope_down) and
            (not c.USE_SLOPE_STRENGTH or strong_trend)
        )

        rsi = row["rsi"]
        rsi_ok_long  = (not c.USE_RSI_FILTER) or (rsi < c.RSI_OVERSOLD)
        rsi_ok_short = (not c.USE_RSI_FILTER) or (rsi > c.RSI_OVERBOUGHT)

        long_allowed  = (self.position_size <= 0) and (self.last_trade != "long")
        short_allowed = (self.position_size >= 0) and (self.last_trade != "short")

        htf_trend = row.get("htf_trend", "flat")
        long_allowed  = long_allowed and (htf_trend == "up")
        short_allowed = short_allowed and (htf_trend == "down")

        long_cond  = raw_long and ema_ok_long and rsi_ok_long and long_allowed
        short_cond = raw_short and ema_ok_short and rsi_ok_short and short_allowed

        if long_cond:
            sig = self._create_signal(row, "BUY")
            sig["strategy_type"] = "doji_breakout"
            return sig
        if short_cond:
            sig = self._create_signal(row, "SELL")
            sig["strategy_type"] = "doji_breakout"
            return sig

        angle9 = row.get("ema9_angle_deg", np.nan)
        bullish_rev = self._is_bullish_reversal_actual(df, i)
        bearish_rev = self._is_bearish_reversal_actual(df, i)

        ema9_long_trend  = angle9 >= c.EMA9_MIN_ANGLE_DEG
        ema9_short_trend = angle9 <= -c.EMA9_MIN_ANGLE_DEG

        long2_allowed  = (self.position_size <= 0) and (self.last_trade != "long") and (htf_trend == "up")
        short2_allowed = (self.position_size >= 0) and (self.last_trade != "short") and (htf_trend == "down")

        long2_cond  = ema9_long_trend  and bullish_rev  and long2_allowed
        short2_cond = ema9_short_trend and bearish_rev and short2_allowed

        if long2_cond:
            sig = self._create_signal(row, "BUY")
            sig["strategy_type"] = "ema9_angle_reversal_actual"
            return sig
        if short2_cond:
            sig = self._create_signal(row, "SELL")
            sig["strategy_type"] = "ema9_angle_reversal_actual"
            return sig

        return None

    def _create_signal(self, row: pd.Series, direction: str) -> Dict:
        und_price = row["und_close"]
        opt_symbol = self._pick_itm_option_symbol(underlying_price=und_price, direction=direction)

        signal = {
            "timestamp": row["timestamp"],
            "index_symbol": self.config.UNDERLYING_SYMBOL,
            "index_price": float(und_price),
            "direction": direction,
            "option_symbol": opt_symbol,
        }

        if direction == "BUY":
            self.position_size = 1
            self.last_trade = "long"
        else:
            self.position_size = -1
            self.last_trade = "short"

        return signal

    # ---------------------- BACKTEST (1-bar confirmation, session filtered) ----------------------

    def run_backtest(self) -> pd.DataFrame:
        try:
            print("🚀 Starting NIFTY Index→ITM Option Backtest (1-bar confirmation, session-filtered)...")
            df_index_1 = self.get_index_data()
            signals = self.generate_signals(df_index_1)

            if signals.empty:
                print("⚠️  No signals generated - adjust strategy parameters or HTF filter")
                return signals

            option_cache: Dict[str, pd.DataFrame] = {}
            for sym in signals["option_symbol"].unique():
                opt_df = self._fetch_option_history(sym, self.config.HIST_DAYS)
                opt_df = opt_df.sort_index()
                option_cache[sym] = opt_df

            pnl_pct_list = []
            pnl_rupees_list = []
            result_list = []
            exit_time_list = []
            exit_price_list = []
            entry_price_list = []

            # 1-bar confirmation: entry on next option bar after signal timestamp
            for _, s in signals.iterrows():
                signal_ts = s["timestamp"]
                opt_symbol = s["option_symbol"]

                opt_df = option_cache[opt_symbol]

                # find index of first bar strictly AFTER signal_ts
                idx_pos = opt_df.index.searchsorted(signal_ts, side="right")
                if idx_pos >= len(opt_df):
                    pnl_pct_list.append(0.0)
                    pnl_rupees_list.append(0.0)
                    result_list.append("unknown")
                    exit_time_list.append(pd.NaT)
                    exit_price_list.append(np.nan)
                    entry_price_list.append(np.nan)
                    continue

                entry_idx = int(idx_pos)
                entry_bar = opt_df.iloc[entry_idx]

                raw_entry_price = float(entry_bar["open"])   # next bar open to mimic live entry. [web:75][web:79]
                # 0.5% slippage premium for ITM option buys. [web:52][web:77]
                entry_price = raw_entry_price * 1.005
                entry_price_list.append(entry_price)

                tp_level = entry_price * (1 + self.config.TAKE_PROFIT_PCT / 100.0)
                sl_level = entry_price * (1 - self.config.STOP_LOSS_PCT / 100.0)

                exit_idx = None
                exit_price = None
                pnl_pct = 0.0
                result = "loss"

                # Exit search: from bar after entry_idx onward
                for j in range(entry_idx + 1, len(opt_df)):
                    bar = opt_df.iloc[j]
                    bar_high = float(bar["high"])
                    bar_low  = float(bar["low"])

                    hit_tp = bar_high >= tp_level
                    hit_sl = bar_low  <= sl_level

                    if hit_tp or hit_sl:
                        exit_idx = j
                        if hit_tp and hit_sl:
                            # conservative: assume SL hit first
                            exit_price = sl_level
                            pnl_pct = (sl_level - entry_price) / entry_price * 100.0
                            result = "loss"
                        elif hit_tp:
                            exit_price = tp_level
                            pnl_pct = (tp_level - entry_price) / entry_price * 100.0
                            result = "profit"
                        else:
                            exit_price = sl_level
                            pnl_pct = (sl_level - entry_price) / entry_price * 100.0
                            result = "loss"
                        break

                if exit_idx is None:
                    last_bar = opt_df.iloc[-1]
                    exit_price = float(last_bar["close"])
                    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                    result = "profit" if pnl_pct > 0 else "loss"
                    exit_time = opt_df.index[-1]
                else:
                    exit_time = opt_df.index[exit_idx]

                pnl_rupees = entry_price * self.config.NIFTY_LOT_SIZE * (pnl_pct / 100.0)

                pnl_pct_list.append(pnl_pct)
                pnl_rupees_list.append(pnl_rupees)
                result_list.append(result)
                exit_time_list.append(exit_time)
                exit_price_list.append(exit_price)

            signals["entry_price_option"]  = entry_price_list
            signals["exit_time"]           = exit_time_list
            signals["exit_price_option"]   = exit_price_list
            signals["pnl_pct"]             = pnl_pct_list
            signals["pnl_rupees_1lot"]     = pnl_rupees_list
            signals["result"]              = result_list
            signals["cum_pnl_rupees_1lot"] = signals["pnl_rupees_1lot"].cumsum()

            total_trades = len(signals)
            total_profit = signals.loc[signals["result"] == "profit", "pnl_rupees_1lot"].sum()
            total_loss   = signals.loc[signals["result"] == "loss",   "pnl_rupees_1lot"].sum()
            net_pnl      = signals["pnl_rupees_1lot"].sum()
            win_rate     = (len(signals[signals["result"] == "profit"]) / total_trades * 100.0) if total_trades > 0 else 0.0

            csv_path = "HA_Doji_and_EMA9Angle_NIFTY_Index_To_ITM_OptionSignals_with_HTF_SessionFiltered_Tuned_1bar_confirm.csv"
            signals.to_csv(csv_path, index=False, encoding="utf-8")

            summary_lines = [
                "",
                "METRICS,VALUE",
                f"Total Trades,{total_trades}",
                f"Total Profit (₹),{total_profit}",
                f"Total Loss (₹),{total_loss}",
                f"Net PnL (₹),{net_pnl}",
                f"Win Rate (%),{win_rate}",
            ]
            with open(csv_path, "a", encoding="utf-8") as f:
                for line in summary_lines:
                    f.write(line + "\n")

            print("📊 Backtest complete; results saved to", csv_path)
            return signals

        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            raise


# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    bot = TradingBot()
    signals = bot.run_backtest()
