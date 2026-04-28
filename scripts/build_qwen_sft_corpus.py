"""Build a small SFT seed corpus for Qwen from Trader parquet data.

The output is JSONL in chat format:
- system message: strict JSON-only trading policy
- user message: current 1-minute candle + ranked top-40 indicator snapshot
- assistant message: target JSON action

The labels here are heuristic, not expert human trades. The goal is to teach
Qwen the observation schema and the action formatting so pydantic parsing is
less error-prone before GRPO.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import MarketFeatures


TOP40_FEATURE_COLUMNS = (
    "log_ret_1",
    "ret_5",
    "ret_15",
    "hl_range_pct",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "sma_10_rel",
    "sma_20_rel",
    "sma_50_rel",
    "ema_12_rel",
    "ema_21_rel",
    "ema_26_rel",
    "wma_20_rel",
    "macd_line_rel",
    "macd_signal_rel",
    "macd_hist_rel",
    "rsi_14",
    "rsi_6",
    "stoch_k",
    "stoch_d",
    "adx_14",
    "di_spread",
    "mom_10_rel",
    "trix_15",
    "atr_rel",
    "bb_pos",
    "bb_width",
    "keltner_pos",
    "donchian_pos",
    "ret_std_20",
    "volume_z",
    "obv_flow",
    "vwap_rel",
    "adl_flow",
    "mfi_14",
    "cmf_20",
    "supertrend_dir",
    "bos_up",
    "bos_down",
)

FEATURE_GLOSSARY = {
    "log_ret_1": "1-bar log return.",
    "ret_5": "5-bar percentage return.",
    "ret_15": "15-bar percentage return.",
    "hl_range_pct": "High-low range as a percentage of price.",
    "body_pct": "Candle body size as a percentage of price.",
    "upper_wick_pct": "Upper wick size as a percentage of price.",
    "lower_wick_pct": "Lower wick size as a percentage of price.",
    "sma_10_rel": "10-period SMA relative to price.",
    "sma_20_rel": "20-period SMA relative to price.",
    "sma_50_rel": "50-period SMA relative to price.",
    "ema_12_rel": "12-period EMA relative to price.",
    "ema_21_rel": "21-period EMA relative to price.",
    "ema_26_rel": "26-period EMA relative to price.",
    "wma_20_rel": "20-period weighted moving average relative to price.",
    "macd_line_rel": "MACD line relative to price.",
    "macd_signal_rel": "MACD signal line relative to price.",
    "macd_hist_rel": "MACD histogram relative to price.",
    "rsi_14": "14-period RSI.",
    "rsi_6": "6-period RSI.",
    "stoch_k": "Stochastic %K.",
    "stoch_d": "Stochastic %D.",
    "adx_14": "14-period ADX trend strength.",
    "di_spread": "Difference between +DI and -DI.",
    "mom_10_rel": "10-bar momentum relative to price.",
    "trix_15": "15-period TRIX.",
    "atr_rel": "ATR relative to price.",
    "bb_pos": "Position within Bollinger Bands.",
    "bb_width": "Bollinger Band width.",
    "keltner_pos": "Position within Keltner Channel.",
    "donchian_pos": "Position within Donchian Channel.",
    "ret_std_20": "20-bar return volatility.",
    "volume_z": "Volume z-score.",
    "obv_flow": "On-balance volume flow.",
    "vwap_rel": "Price relative to VWAP.",
    "adl_flow": "Accumulation/distribution line flow.",
    "mfi_14": "14-period money flow index.",
    "cmf_20": "20-period Chaikin money flow.",
    "supertrend_dir": "Supertrend direction signal.",
    "bos_up": "Break of structure upward flag.",
    "bos_down": "Break of structure downward flag.",
}

SYSTEM_PROMPT = (
    "You are a crypto trading policy trained on 1-minute candle data. Return only valid JSON with keys "
    "position, take_profit_price, stop_loss_price. "
    "Your objective is to maximize profit. "
    "position must be one of long, short, noop."
)


def iter_parquet_files(data_root: Path) -> Iterable[Path]:
    for path in sorted(data_root.glob("*.parquet")):
        if path.is_file():
            yield path


def make_target_action(position: str, close_price: float, tp_pct: float, sl_pct: float) -> dict[str, float | str]:
    if position == "long":
        return {
            "position": "long",
            "take_profit_price": round(close_price * (1.0 + tp_pct), 8),
            "stop_loss_price": round(close_price * (1.0 - sl_pct), 8),
        }
    if position == "short":
        return {
            "position": "short",
            "take_profit_price": round(close_price * (1.0 - tp_pct), 8),
            "stop_loss_price": round(close_price * (1.0 + sl_pct), 8),
        }
    return {"position": "noop", "take_profit_price": 0.0, "stop_loss_price": 0.0}


def label_action(df: pd.DataFrame, idx: int, horizon: int, long_threshold: float, short_threshold: float) -> str:
    current_close = float(df.iloc[idx]["close"])
    future_close = float(df.iloc[idx + horizon]["close"])
    future_return = (future_close / current_close) - 1.0
    if future_return >= long_threshold:
        return "long"
    if future_return <= -short_threshold:
        return "short"
    return "noop"


def simulate_outcome(df: pd.DataFrame, idx: int, position: str, tp_pct: float, sl_pct: float, max_holding: int) -> dict:
    """Scan forward from idx and report whether TP/SL was hit and realized return.

    Also compute MFE (max favorable excursion) and MAE (max adverse excursion).
    Returns absolute pct values for mfe/mae.
    """
    entry_price = float(df.iloc[idx]["close"])
    if position == "noop":
        return {
            "outcome": "noop",
            "exit_price": entry_price,
            "realized_return": 0.0,
            "holding_bars": 0,
            "exit_idx": idx,
            "mfe": 0.0,
            "mae": 0.0,
        }

    if position == "long":
        take_profit = entry_price * (1.0 + tp_pct)
        stop_loss = entry_price * (1.0 - sl_pct)
    else:
        take_profit = entry_price * (1.0 - tp_pct)
        stop_loss = entry_price * (1.0 + sl_pct)

    start = idx + 1
    end = min(len(df) - 1, idx + max_holding)
    exit_price = float(df.iloc[end]["close"])
    exit_idx = end
    outcome = "timeout"

    mfe = 0.0
    mae = 0.0

    for j in range(start, end + 1):
        row = df.iloc[j]
        high = float(row["high"])
        low = float(row["low"])

        if position == "long":
            mfe = max(mfe, (high / entry_price) - 1.0)
            mae = max(mae, (entry_price / low) - 1.0)
            if low <= stop_loss:
                exit_price = stop_loss
                exit_idx = j
                outcome = "stop_loss"
                break
            if high >= take_profit:
                exit_price = take_profit
                exit_idx = j
                outcome = "take_profit"
                break
        else:
            mfe = max(mfe, (entry_price / low) - 1.0)
            mae = max(mae, (high / entry_price) - 1.0)
            if high >= stop_loss:
                exit_price = stop_loss
                exit_idx = j
                outcome = "stop_loss"
                break
            if low <= take_profit:
                exit_price = take_profit
                exit_idx = j
                outcome = "take_profit"
                break

    if position == "long":
        realized = (exit_price / entry_price) - 1.0
    else:
        realized = (entry_price / exit_price) - 1.0

    return {
        "outcome": outcome,
        "exit_price": round(float(exit_price), 8),
        "realized_return": float(realized),
        "holding_bars": int(exit_idx - idx),
        "exit_idx": int(exit_idx),
        "mfe": float(mfe),
        "mae": float(mae),
    }


def build_prompt(row: pd.Series, symbol: str) -> dict:
    candle = {
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row.get("volume", 0.0)),
        "quote_asset_volume": float(row.get("quote_asset_volume", 0.0)),
        "num_trades": float(row.get("num_trades", 0.0)),
        "taker_buy_base": float(row.get("taker_buy_base", 0.0)),
        "taker_buy_quote": float(row.get("taker_buy_quote", 0.0)),
    }
    features = {name: float(row[name]) for name in TOP40_FEATURE_COLUMNS}
    return {
        "symbol": symbol,
        "timeframe": "1m",
        "candle": candle,
        "features": features,
        "feature_descriptions": {name: FEATURE_GLOSSARY[name] for name in TOP40_FEATURE_COLUMNS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL seed corpus for Qwen SFT.")
    parser.add_argument("--data-root", type=str, default="server/data_trade_episodes/rl_features_by_symbol")
    parser.add_argument("--output", type=str, default="sft_corpus/qwen_trader_seed.jsonl")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-examples-per-symbol", type=int, default=1500)
    parser.add_argument("--long-threshold", type=float, default=0.004)
    parser.add_argument("--short-threshold", type=float, default=0.004)
    parser.add_argument("--take-profit-pct", type=float, default=0.012)
    parser.add_argument("--stop-loss-pct", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance", action="store_true", help="Balance classes per-symbol by downsampling majority class (noop)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for parquet_path in iter_parquet_files(data_root):
            required = ["open", "high", "low", "close", *TOP40_FEATURE_COLUMNS]
            # read only the required columns to save memory
            try:
                df = pd.read_parquet(parquet_path, columns=required)
            except Exception:
                # fallback to full read if parquet engine doesn't support columns
                df = pd.read_parquet(parquet_path)
            missing = [col for col in required if col not in df.columns]
            if missing:
                continue

            # downcast numeric columns to float32 to reduce memory footprint
            for c in required:
                if c in df.columns:
                    try:
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
                    except Exception:
                        pass

            clean_df = df.dropna(subset=required).reset_index(drop=True)
            if len(clean_df) <= args.horizon:
                continue

            symbol = parquet_path.stem

            if not args.balance:
                # stream-write (low memory): cap per-symbol by symbol_written
                symbol_written = 0
                for idx in range(0, len(clean_df) - args.horizon):
                    if symbol_written >= args.max_examples_per_symbol:
                        break
                    row = clean_df.iloc[idx]
                    current_close = float(row["close"])
                    position = label_action(clean_df, idx, args.horizon, args.long_threshold, args.short_threshold)
                    target = make_target_action(position, current_close, args.take_profit_pct, args.stop_loss_pct)
                    user_payload = build_prompt(row, symbol)
                    ground = simulate_outcome(clean_df, idx, position, args.take_profit_pct, args.stop_loss_pct, args.horizon)
                    assistant_payload = {"action": target, "ground_truth": ground}
                    record = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
                            {"role": "assistant", "content": json.dumps(assistant_payload, separators=(",", ":"))},
                        ]
                    }
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
                    written += 1
                    symbol_written += 1
            else:
                # reservoir-sample per-class to cap memory while balancing
                rnd = random.Random(args.seed + (abs(hash(symbol)) % 10_000))
                capacity = max(1, args.max_examples_per_symbol // 3)
                reservoirs: dict[str, list] = {"long": [], "short": [], "noop": []}
                seen: dict[str, int] = {"long": 0, "short": 0, "noop": 0}

                for idx in range(0, len(clean_df) - args.horizon):
                    row = clean_df.iloc[idx]
                    current_close = float(row["close"])
                    position = label_action(clean_df, idx, args.horizon, args.long_threshold, args.short_threshold)
                    target = make_target_action(position, current_close, args.take_profit_pct, args.stop_loss_pct)
                    user_payload = build_prompt(row, symbol)
                    ground = simulate_outcome(clean_df, idx, position, args.take_profit_pct, args.stop_loss_pct, args.horizon)
                    assistant_payload = {"action": target, "ground_truth": ground}
                    record = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
                            {"role": "assistant", "content": json.dumps(assistant_payload, separators=(",", ":"))},
                        ]
                    }

                    cls = position if position in reservoirs else "noop"
                    seen[cls] += 1
                    if len(reservoirs[cls]) < capacity:
                        reservoirs[cls].append(record)
                    else:
                        i = rnd.randrange(seen[cls])
                        if i < capacity:
                            reservoirs[cls][i] = record

                # combine reservoirs and write shuffled
                combined = []
                for k in ("long", "short", "noop"):
                    combined.extend(reservoirs[k])
                rnd.shuffle(combined)
                for rec in combined:
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                    written += 1

    print(f"Wrote {written} examples to {out_path}")


if __name__ == "__main__":
    main()
