# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trader environment implementation for RL training on candle features."""

from pathlib import Path
from random import Random
from uuid import uuid4

import pandas as pd
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MarketFeatures, TraderAction, TraderObservation
except ImportError:
    from models import MarketFeatures, TraderAction, TraderObservation

class TraderEnvironment(Environment):
    """Multi-episode trading environment built from per-coin feature data."""

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        episode_length: int = 1024,
        max_holding_bars: int = 60,
        slippage_bps: float = 5.0,
        drawdown_penalty: float = 0.10,
        default_take_profit_pct: float = 0.012,
        default_stop_loss_pct: float = 0.006,
        seed: int = 42,
    ):
        """Initialize the trader environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._rng = Random(seed)
        self._episode_length = episode_length
        self._max_holding_bars = max(1, max_holding_bars)
        self._slippage_rate = slippage_bps / 10000.0
        self._drawdown_penalty = drawdown_penalty
        self._default_take_profit_pct = default_take_profit_pct
        self._default_stop_loss_pct = default_stop_loss_pct

        self._data_root = Path(__file__).resolve().parent / "data_trade_episodes" / "rl_features_by_symbol"
        self._coin_dfs: dict[str, pd.DataFrame] = self._load_coin_data()
        if not self._coin_dfs:
            raise ValueError(f"No valid coin data found in {self._data_root}")

        self._symbol = ""
        self._df = pd.DataFrame()
        self._idx = 0
        self._episode_end_idx = 0
        self._position: str = "none"
        self._entry_price: float | None = None
        self._take_profit_price: float | None = None
        self._stop_loss_price: float | None = None
        self._pnl: float = 0.0
        self._peak_pnl: float = 0.0
        self._drawdown: float = 0.0
        self._slippage_paid: float = 0.0

    def _load_coin_data(self) -> dict[str, pd.DataFrame]:
        """Load per-coin dataframes, keeping only rows valid for RL observations."""
        coin_data: dict[str, pd.DataFrame] = {}
        required = list(MarketFeatures.FEATURE_COLUMNS) + ["open", "high", "low", "close"]

        for file_path in sorted(self._data_root.glob("*.parquet")):
            symbol = file_path.stem
            df = pd.read_parquet(file_path)

            missing = [col for col in required if col not in df.columns]
            if missing:
                continue

            clean_df = df.dropna(subset=required).reset_index(drop=True)
            if len(clean_df) < max(3, self._episode_length // 2):
                continue

            coin_data[symbol] = clean_df

        return coin_data

    def _position_sign(self) -> int:
        if self._position == "long":
            return 1
        if self._position == "short":
            return -1
        return 0

    def _mark_to_market_pnl(self, price: float) -> float:
        if self._entry_price is None or self._position == "none":
            return 0.0
        if self._position == "long":
            return (price / self._entry_price) - 1.0
        return (self._entry_price / price) - 1.0

    def _effective_tp_sl(self, position: str, reference_price: float, action: TraderAction) -> tuple[float | None, float | None]:
        tp = action.take_profit_price if action.take_profit_price > 0 else 0.0
        sl = action.stop_loss_price if action.stop_loss_price > 0 else 0.0

        if position == "long":
            take_profit = tp if tp > 0 else reference_price * (1.0 + self._default_take_profit_pct)
            stop_loss = sl if sl > 0 else reference_price * (1.0 - self._default_stop_loss_pct)
            return take_profit, stop_loss

        if position == "short":
            take_profit = tp if tp > 0 else reference_price * (1.0 - self._default_take_profit_pct)
            stop_loss = sl if sl > 0 else reference_price * (1.0 + self._default_stop_loss_pct)
            return take_profit, stop_loss

        return None, None

    def _build_observation(self, *, done: bool, reward: float) -> TraderObservation:
        """Create observation from current timestep."""
        row = self._df.iloc[self._idx]
        market_features = MarketFeatures.from_row(row)
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        current_price = float(row["close"])

        return TraderObservation(
            market_features=market_features,
            symbol=self._symbol,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            current_price=current_price,
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "idx": self._idx,
            },
        )

    def _simulate_trade(self, action: TraderAction) -> float:
        """Score one action by scanning forward until TP/SL or horizon exit."""
        if action.position == "noop":
            return 0.0

        entry_price = float(self._df.iloc[self._idx]["close"])
        take_profit_price, stop_loss_price = self._effective_tp_sl(action.position, entry_price, action)
        assert take_profit_price is not None and stop_loss_price is not None

        start_idx = self._idx + 1
        end_idx = min(self._episode_end_idx, self._idx + 1 + self._max_holding_bars)
        exit_price = float(self._df.iloc[max(start_idx, min(end_idx - 1, len(self._df) - 1))]["close"])
        worst_adverse = 0.0

        for look_idx in range(start_idx, end_idx):
            row = self._df.iloc[look_idx]
            high = float(row["high"])
            low = float(row["low"])

            if action.position == "long":
                worst_adverse = max(worst_adverse, max(0.0, (entry_price - low) / entry_price))
                if low <= stop_loss_price:
                    exit_price = stop_loss_price
                    break
                if high >= take_profit_price:
                    exit_price = take_profit_price
                    break
            else:
                worst_adverse = max(worst_adverse, max(0.0, (high - entry_price) / entry_price))
                if high >= stop_loss_price:
                    exit_price = stop_loss_price
                    break
                if low <= take_profit_price:
                    exit_price = take_profit_price
                    break

        if action.position == "long":
            realized_return = (exit_price / entry_price) - 1.0
        else:
            realized_return = (entry_price / exit_price) - 1.0

        # Keep reward components unit-consistent: realized_return is fractional,
        # so slippage must also be expressed as a fractional round-trip cost.
        slippage_cost = 2.0 * self._slippage_rate
        return realized_return - slippage_cost - (self._drawdown_penalty * worst_adverse)

    def _apply_action(self, action: TraderAction, current_price: float) -> float:
        """Apply action and return slippage cost."""
        if action.position == "noop":
            return 0.0

        prev_sign = self._position_sign()
        desired_sign = 1 if action.position == "long" else -1
        slippage_cost = abs(desired_sign - prev_sign) * current_price * self._slippage_rate

        self._position = action.position
        self._entry_price = current_price
        self._take_profit_price, self._stop_loss_price = self._effective_tp_sl(action.position, current_price, action)
        self._slippage_paid += slippage_cost
        return slippage_cost

    def _exit_position(self) -> None:
        self._position = "none"
        self._entry_price = None
        self._take_profit_price = None
        self._stop_loss_price = None

    def _barrier_exit_price(self, next_row: pd.Series) -> float | None:
        if self._position == "long":
            if self._stop_loss_price is not None and float(next_row["low"]) <= self._stop_loss_price:
                return self._stop_loss_price
            if self._take_profit_price is not None and float(next_row["high"]) >= self._take_profit_price:
                return self._take_profit_price
        elif self._position == "short":
            if self._stop_loss_price is not None and float(next_row["high"]) >= self._stop_loss_price:
                return self._stop_loss_price
            if self._take_profit_price is not None and float(next_row["low"]) <= self._take_profit_price:
                return self._take_profit_price
        return None

    def reset(self) -> TraderObservation:
        """Reset the environment and sample a fresh episode."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._position = "none"
        self._entry_price = None
        self._take_profit_price = None
        self._stop_loss_price = None

        self._symbol = self._rng.choice(list(self._coin_dfs.keys()))
        self._df = self._coin_dfs[self._symbol]

        max_start = max(0, len(self._df) - self._episode_length - 1)
        start_idx = self._rng.randint(0, max_start) if max_start > 0 else 0
        self._idx = start_idx
        self._episode_end_idx = min(len(self._df) - 1, start_idx + self._episode_length)

        return self._build_observation(done=False, reward=0.0)

    def step(self, action: TraderAction) -> TraderObservation:  # type: ignore[override]
        """Execute one trading step and return the next candle observation."""
        self._state.step_count += 1

        is_terminal_now = self._idx >= self._episode_end_idx - 1
        if is_terminal_now:
            return self._build_observation(done=True, reward=0.0)

        reward = self._simulate_trade(action)
        self._idx = min(self._idx + 1, self._episode_end_idx - 1)
        done = self._idx >= self._episode_end_idx - 1

        return self._build_observation(done=done, reward=reward)

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
if __name__ == "__main__":
    env = TraderEnvironment()
    env.reset()