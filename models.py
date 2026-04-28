# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Trader environment."""

from typing import Any, ClassVar, Literal, Mapping

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class MarketFeatures(BaseModel):
    """Normalized/engineered feature vector for RL observation."""

    FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
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
        "stoch_cross",
        "adx_14",
        "di_spread",
        "mom_10_rel",
        "trix_15",
        "dpo_20_rel",
        "atr_rel",
        "bb_pos",
        "bb_width",
        "keltner_pos",
        "keltner_width",
        "donchian_pos",
        "ret_std_20",
        "chaikin_vol",
        "vol_stop_dist",
        "volume_z",
        "obv_flow",
        "vwap_rel",
        "adl_flow",
        "mfi_14",
        "eom_14",
        "vol_osc",
        "cmf_20",
        "ema_cross_5_8",
        "ema_cross_8_13",
        "scalp_rsi_6",
        "bb_bounce",
        "macd_flip",
        "supertrend_dir",
        "bos_up",
        "bos_down",
        "mss_proxy",
        "liq_sweep_high",
        "liq_sweep_low",
        "equal_highs",
        "equal_lows",
        "fvg_up",
        "fvg_down",
        "fvg_gap_rel",
        "premium_discount",
        "dist_support",
        "dist_resistance",
        "channel_pos",
        "fib_618_dist",
        "dist_pivot",
        "dist_r1",
        "dist_s1",
        "taker_buy_ratio",
        "trade_intensity",
        "quote_per_trade",
    )

    log_ret_1: float = 0.0
    ret_5: float = 0.0
    ret_15: float = 0.0
    hl_range_pct: float = 0.0
    body_pct: float = 0.0
    upper_wick_pct: float = 0.0
    lower_wick_pct: float = 0.0
    sma_10_rel: float = 0.0
    sma_20_rel: float = 0.0
    sma_50_rel: float = 0.0
    ema_12_rel: float = 0.0
    ema_21_rel: float = 0.0
    ema_26_rel: float = 0.0
    wma_20_rel: float = 0.0
    macd_line_rel: float = 0.0
    macd_signal_rel: float = 0.0
    macd_hist_rel: float = 0.0
    rsi_14: float = 0.0
    rsi_6: float = 0.0
    stoch_k: float = 0.0
    stoch_d: float = 0.0
    stoch_cross: float = 0.0
    adx_14: float = 0.0
    di_spread: float = 0.0
    mom_10_rel: float = 0.0
    trix_15: float = 0.0
    dpo_20_rel: float = 0.0
    atr_rel: float = 0.0
    bb_pos: float = 0.0
    bb_width: float = 0.0
    keltner_pos: float = 0.0
    keltner_width: float = 0.0
    donchian_pos: float = 0.0
    ret_std_20: float = 0.0
    chaikin_vol: float = 0.0
    vol_stop_dist: float = 0.0
    volume_z: float = 0.0
    obv_flow: float = 0.0
    vwap_rel: float = 0.0
    adl_flow: float = 0.0
    mfi_14: float = 0.0
    eom_14: float = 0.0
    vol_osc: float = 0.0
    cmf_20: float = 0.0
    ema_cross_5_8: float = 0.0
    ema_cross_8_13: float = 0.0
    scalp_rsi_6: float = 0.0
    bb_bounce: float = 0.0
    macd_flip: float = 0.0
    supertrend_dir: float = 0.0
    bos_up: float = 0.0
    bos_down: float = 0.0
    mss_proxy: float = 0.0
    liq_sweep_high: float = 0.0
    liq_sweep_low: float = 0.0
    equal_highs: float = 0.0
    equal_lows: float = 0.0
    fvg_up: float = 0.0
    fvg_down: float = 0.0
    fvg_gap_rel: float = 0.0
    premium_discount: float = 0.0
    dist_support: float = 0.0
    dist_resistance: float = 0.0
    channel_pos: float = 0.0
    fib_618_dist: float = 0.0
    dist_pivot: float = 0.0
    dist_r1: float = 0.0
    dist_s1: float = 0.0
    taker_buy_ratio: float = 0.0
    trade_intensity: float = 0.0
    quote_per_trade: float = 0.0

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "MarketFeatures":
        """Build features from a dataframe row-like mapping."""

        data = {name: float(row[name]) for name in cls.FEATURE_COLUMNS}
        return cls(**data)



class TraderAction(Action):
    """Explicit trading action with optional risk levels."""

    position: Literal["long", "short", "noop"]
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0

class TraderObservation(Observation):
    """Observation for the Trader environment."""

    market_features: MarketFeatures = Field(
        default_factory=MarketFeatures,
        description="Normalized engineered market features for policy input",
    )
    symbol: str = ""
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    current_price: float = 0.0
