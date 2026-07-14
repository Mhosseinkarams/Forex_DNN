import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime

from ML.feature_registry import FeatureRegistry
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, StructureLevel, Zone, BOS, CHOCH
from Market_Data_Pipeline.dataset_types import FeatureVector

logger = logging.getLogger("FeaturePipeline")

class FeaturePipeline:
    """
    Purpose:
        Registry-driven Feature Pipeline converting raw indicators, market structure,
        and supply/demand information into clean, normalized feature vectors.
        Contains NO hardcoded feature lists in the extraction loop; instead, it
        dynamically queries the Feature Registry for enabled features and computes
        them using registered mapping handlers.
    """
    def __init__(self, registry: Optional[FeatureRegistry] = None):
        self.registry = registry or FeatureRegistry()
        self._setup_extractors()
        # Initialize an instance-level cache to prevent redundant re-computation of features
        self._cache = {}

    def _setup_extractors(self):
        """
        Map every feature name from the Feature Registry to its extractor function.
        This isolates the extraction logic and prevents hardcoding feature names in loops.
        """
        self._extractors = {
            "ema50_slope": self._extract_ema50_slope,
            "ema600_slope": self._extract_ema600_slope,
            "ema800_slope": self._extract_ema800_slope,
            "ema_separation": self._extract_ema_separation,
            "ema_compression": self._extract_ema_compression,
            "distance_to_ema50": self._extract_distance_to_ema50,
            "distance_to_ema600": self._extract_distance_to_ema600,
            "distance_to_ema800": self._extract_distance_to_ema800,
            "candle_body": self._extract_candle_body,
            "upper_wick": self._extract_upper_wick,
            "lower_wick": self._extract_lower_wick,
            "bos_count_last_n": self._extract_bos_count_last_n,
            "choch_count_last_n": self._extract_choch_count_last_n,
            "time_since_last_bos": self._extract_time_since_last_bos,
            "time_since_last_choch": self._extract_time_since_last_choch,
            "bos_direction": self._extract_bos_direction,
            "choch_direction": self._extract_choch_direction,
            "protected_high_distance": self._extract_protected_high_distance,
            "protected_low_distance": self._extract_protected_low_distance,
            "supply_distance": self._extract_supply_distance,
            "supply_width": self._extract_supply_width,
            "supply_strength": self._extract_supply_strength,
            "supply_freshness": self._extract_supply_freshness,
            "supply_touch_count": self._extract_supply_touch_count,
            "demand_distance": self._extract_demand_distance,
            "demand_width": self._extract_demand_width,
            "demand_strength": self._extract_demand_strength,
            "demand_freshness": self._extract_demand_freshness,
            "demand_touch_count": self._extract_demand_touch_count,
            "candle_range": self._extract_candle_range,
            "volume": self._extract_volume,
            "spread": self._extract_spread,
            "atr": self._extract_atr,
            "atr_percentile": self._extract_atr_percentile,
            "atr_ratio": self._extract_atr_ratio,
            "realized_volatility": self._extract_realized_volatility,
            "rolling_std": self._extract_rolling_std,
            "hour": self._extract_hour,
            "weekday": self._extract_weekday,
            "session": self._extract_session,
            "trend_score": self._extract_trend_score,
            "range_score": self._extract_range_score,
            "compression_score": self._extract_compression_score,
            "distance_to_nearest_high": self._extract_distance_to_nearest_high,
            "distance_to_nearest_low": self._extract_distance_to_nearest_low,
            "distance_to_structure_break": self._extract_distance_to_structure_break,
            "distance_to_invalidation_level": self._extract_distance_to_invalidation_level,
            "risk_reward_estimate": self._extract_risk_reward_estimate,
            "ema50_distance_v1": self._extract_ema50_distance_v1,
            "ema50_distance_v2": self._extract_ema50_distance_v2,
        }

    def clear_cache(self) -> None:
        """Clear the calculated feature cache."""
        self._cache.clear()

    def extract_all(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int = -1) -> Dict[str, Any]:
        """
        Extract features for a specific row index (defaulting to the last bar).
        Ensures strict point-in-time calculation (only looking back from idx to avoid leakage).
        """
        if idx < 0:
            idx = len(df) + idx

        # Ensure we are within range
        idx = max(0, min(idx, len(df) - 1))

        # Check Cache to optimize runtime performance during live trading or sequential backtests
        cache_key = (id(df), idx)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        features = {}
        enabled = self.registry.list_enabled()

        # Iterate over only the enabled registry definitions
        for f in enabled:
            name = f.name
            if name in self._extractors:
                try:
                    val = self._extractors[name](df, msg, idx)
                    # Support registry default_value fallback if NaN or None is returned
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        val = f.default_value
                    features[name] = val
                except Exception as e:
                    logger.debug(f"Error extracting feature {name} at index {idx}: {e}")
                    features[name] = f.default_value
            else:
                logger.warning(f"No extractor handler defined for registered feature: '{name}'")
                features[name] = f.default_value

        self._cache[cache_key] = features.copy()
        return features

    def extract_runtime(
        self,
        df: pd.DataFrame,
        msg: MarketStructureGraph,
        idx: int = -1,
        indicator_df: Optional[pd.DataFrame] = None,
        account_session_context: Optional[Dict[str, Any]] = None,
        strategy_context: Optional[Dict[str, Any]] = None,
        return_df: bool = False
    ) -> Union[FeatureVector, pd.DataFrame]:
        """
        Generate exactly the same features used during training from live market data.
        Verifies and validates the generated feature vectors for missing values, NaNs,
        infinites, invalid types, and unexpected columns.
        """
        # Resolve the index
        if idx < 0:
            idx = len(df) + idx
        idx = max(0, min(idx, len(df) - 1))

        # Handle indicator_df merge if supplied separate from the main dataframe
        if indicator_df is not None:
            combined_df = df.copy()
            for col in indicator_df.columns:
                if col not in combined_df.columns:
                    combined_df[col] = indicator_df[col]
            df_to_use = combined_df
        else:
            df_to_use = df

        # Base feature extraction
        extracted = self.extract_all(df_to_use, msg, idx=idx)

        # Apply account/session context if passed
        if account_session_context:
            for k, v in account_session_context.items():
                if k in extracted:
                    extracted[k] = v
                # support direct extraction keys like 'spread' or 'session'
                elif k.lower() == "session" and "session" in extracted:
                    extracted["session"] = v
                elif k.lower() == "spread" and "spread" in extracted:
                    extracted["spread"] = v

        # Apply strategy context if passed
        if strategy_context:
            for k, v in strategy_context.items():
                if k in extracted:
                    extracted[k] = v

        # VALIDATION STAGE
        enabled = self.registry.list_enabled()
        enabled_names = {f.name for f in enabled}

        warnings = []
        failures = []

        # 1. Unexpected columns detection
        for name in list(extracted.keys()):
            if name not in enabled_names:
                failures.append(f"Unexpected feature column extracted: '{name}'")
                logger.warning(f"Validation Failure: Unexpected column '{name}' is present in feature vector.")

        # 2. Missing columns and value verification
        cleaned_features = {}
        for f in enabled:
            name = f.name

            # Retrieve value or set default if missing
            if name not in extracted:
                failures.append(f"Missing feature: '{name}'")
                logger.error(f"Validation Failure: Feature '{name}' is missing. Backing up to default: {f.default_value}")
                val = f.default_value
            else:
                val = extracted[name]

            # Detect NaNs
            if val is None or (isinstance(val, float) and np.isnan(val)):
                failures.append(f"NaN value in feature '{name}'")
                logger.error(f"Validation Failure: NaN detected in feature '{name}'. Backing up to default: {f.default_value}")
                val = f.default_value

            # Detect Infinite values
            if isinstance(val, float) and np.isinf(val):
                failures.append(f"Infinite value in feature '{name}'")
                logger.error(f"Validation Failure: Infinite value detected in feature '{name}'. Backing up to default: {f.default_value}")
                val = f.default_value

            # Standardize types and coerce
            expected_dtype = f.dtype
            type_map = {
                "float": float,
                "int": int,
                "str": str,
                float: float,
                int: int,
                str: str
            }
            target_type = type_map.get(expected_dtype, float)

            if not isinstance(val, target_type):
                try:
                    coerced = target_type(val)
                    logger.debug(f"Coerced feature '{name}' from type {type(val)} to {target_type} successfully.")
                    val = coerced
                except Exception as e:
                    failures.append(f"Type mismatch / Coercion failure in feature '{name}': {e}")
                    logger.error(f"Validation Failure: Type coercion failed for feature '{name}' ({type(val)} -> {target_type}). Backing up to default: {f.default_value}")
                    val = f.default_value

            cleaned_features[name] = val

        # Guarantee strict feature ordering defined strictly by the registry
        ordered_features = {f.name: cleaned_features[f.name] for f in enabled}
        ordered_values = [ordered_features[f.name] for f in enabled]
        vector_array = np.array(ordered_values, dtype=object)

        # Assemble metadata
        meta = {
            "symbol": msg.symbol,
            "timeframe": msg.timeframe,
            "timestamp": str(msg.timestamp),
            "failures": failures,
            "feature_hash": self.registry.compute_hash()
        }

        # Return requested type
        if return_df:
            # Single-row Pandas DataFrame
            return pd.DataFrame([ordered_features])
        else:
            return FeatureVector(
                features=ordered_features,
                vector=vector_array,
                metadata=meta
            )

    # --- FEATURE EXTRACTORS ---

    def _extract_ema50_slope(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "ema_slope_50" in row:
            return float(row["ema_slope_50"])
        # Fallback manual calculation of slope: dy over 32 bars normalized by ATR
        ema_col = "ema_50"
        atr_col = "atr_14"
        if ema_col in df.columns and atr_col in df.columns and idx >= 32:
            dy = df.iloc[idx][ema_col] - df.iloc[idx - 32][ema_col]
            atr = df.iloc[idx][atr_col]
            return float(abs(dy) / (atr + 1e-9))
        return 0.0

    def _extract_ema600_slope(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "ema_slope_600" in row:
            return float(row["ema_slope_600"])
        return 0.0

    def _extract_ema800_slope(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "ema_slope_800" in row:
            return float(row["ema_slope_800"])
        return 0.0

    def _extract_ema_separation(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "ema_50" in row:
            fast = row["ema_50"]
            slow = row.get("ema_600", row.get("ema_800", fast))
            return float(abs(fast - slow) * 10000.0)  # In pips
        return 0.0

    def _extract_ema_compression(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        sep = self._extract_ema_separation(df, msg, idx)
        return float(1.0 / (1.0 + sep))

    def _extract_distance_to_ema50(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "dist_ema_50" in row:
            return float(row["dist_ema_50"])
        return 0.0

    def _extract_distance_to_ema600(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "dist_ema_600" in row:
            return float(row["dist_ema_600"])
        return 0.0

    def _extract_distance_to_ema800(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "dist_ema_800" in row:
            return float(row["dist_ema_800"])
        return 0.0

    def _extract_candle_body(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "body_pct" in row:
            return float(row["body_pct"])
        return 0.0

    def _extract_upper_wick(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "upper_shadow" in row and "candle_size" in row:
            return float(row["upper_shadow"] / (row["candle_size"] + 1e-9))
        return 0.0

    def _extract_lower_wick(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "lower_shadow" in row and "candle_size" in row:
            return float(row["lower_shadow"] / (row["candle_size"] + 1e-9))
        return 0.0

    def _extract_bos_count_last_n(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> int:
        # Count BOS in last 100 bars from idx
        n = 100
        count = 0
        for b in msg.bos:
            if idx - n <= b.index <= idx:
                count += 1
        return count

    def _extract_choch_count_last_n(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> int:
        n = 100
        count = 0
        for c in msg.choch:
            if idx - n <= c.index <= idx:
                count += 1
        return count

    def _extract_time_since_last_bos(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        last_b = None
        for b in msg.bos:
            if b.index <= idx:
                if last_b is None or b.index > last_b.index:
                    last_b = b
        if last_b:
            return float(idx - last_b.index)
        return 999.0

    def _extract_time_since_last_choch(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        last_c = None
        for c in msg.choch:
            if c.index <= idx:
                if last_c is None or c.index > last_c.index:
                    last_c = c
        if last_c:
            return float(idx - last_c.index)
        return 999.0

    def _extract_bos_direction(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> int:
        last_b = None
        for b in msg.bos:
            if b.index <= idx:
                if last_b is None or b.index > last_b.index:
                    last_b = b
        return last_b.direction if last_b else 0

    def _extract_choch_direction(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> int:
        last_c = None
        for c in msg.choch:
            if c.index <= idx:
                if last_c is None or c.index > last_c.index:
                    last_c = c
        if last_c:
            return 1 if last_c.new_trend > last_c.previous_trend else -1
        return 0

    def _extract_protected_high_distance(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        close = row["Close"]
        atr = row.get("atr_14", 0.0001)
        highs = [s.price for s in msg.swing_highs if s.confirmation_candle <= idx]
        if highs:
            return float((max(highs) - close) / atr)
        return -1.0

    def _extract_protected_low_distance(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        close = row["Close"]
        atr = row.get("atr_14", 0.0001)
        lows = [s.price for s in msg.swing_lows if s.confirmation_candle <= idx]
        if lows:
            return float((close - min(lows)) / atr)
        return -1.0

    def _extract_supply_distance(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        distance = row.get("nearest_supply_distance", np.nan)
        atr = row.get("atr_14", 0.0001)
        return float(distance / atr) if pd.notna(distance) else -1.0

    def _extract_supply_width(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        close = row["Close"]
        active_supplies = [z for z in msg.supply_zones if (z.created_idx <= idx and (not z.broken or z.broken_idx > idx))]
        valid_supplies = [z for z in active_supplies if z.lower > close]
        if valid_supplies:
            nearest = min(valid_supplies, key=lambda z: z.lower)
            return float(nearest.width * 10000.0)  # Width in pips
        return 0.0

    def _extract_supply_strength(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        return float(df.iloc[idx].get("supply_strength", 0.0))

    def _extract_supply_freshness(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> int:
        return int(df.iloc[idx].get("supply_freshness", 0))

    def _extract_supply_touch_count(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> int:
        return int(df.iloc[idx].get("supply_touch_count", 0))

    def _extract_demand_distance(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        distance = row.get("nearest_demand_distance", np.nan)
        atr = row.get("atr_14", 0.0001)
        return float(distance / atr) if pd.notna(distance) else -1.0

    def _extract_demand_width(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        close = row["Close"]
        active_demands = [z for z in msg.demand_zones if (z.created_idx <= idx and (not z.broken or z.broken_idx > idx))]
        valid_demands = [z for z in active_demands if z.upper < close]
        if valid_demands:
            nearest = max(valid_demands, key=lambda z: z.upper)
            return float(nearest.width * 10000.0)
        return 0.0

    def _extract_demand_strength(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        return float(df.iloc[idx].get("demand_strength", 0.0))

    def _extract_demand_freshness(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> int:
        return int(df.iloc[idx].get("demand_freshness", 0))

    def _extract_demand_touch_count(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> int:
        return int(df.iloc[idx].get("demand_touch_count", 0))

    def _extract_candle_range(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        atr = row.get("atr_14", 0.0001)
        return float((row["High"] - row["Low"]) / atr)

    def _extract_volume(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        return float(df.iloc[idx].get("TickVolume", 0.0))

    def _extract_spread(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        return float(df.iloc[idx].get("Spread", 0.0))

    def _extract_atr(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        return float(df.iloc[idx].get("atr_14", 0.0001) * 10000.0)  # In pips

    def _extract_atr_percentile(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        # Calculate atr rolling percentile over last 200 bars
        atr_col = "atr_14"
        if atr_col in df.columns:
            start_idx = max(0, idx - 200)
            window = df.iloc[start_idx:idx+1][atr_col]
            if len(window) > 1:
                val = df.iloc[idx][atr_col]
                pct = (window < val).sum() / len(window)
                return float(pct)
        return 0.5

    def _extract_atr_ratio(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        atr_col = "atr_14"
        if atr_col in df.columns and idx >= 50:
            fast = df.iloc[idx][atr_col]
            # Simple average of atr_14 over 50 bars as slow ATR proxy
            slow = df.iloc[idx-50:idx+1][atr_col].mean()
            return float(fast / (slow + 1e-9))
        return 1.0

    def _extract_realized_volatility(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        if idx >= 20:
            closes = df.iloc[idx-20:idx+1]["Close"]
            returns = np.log(closes / closes.shift(1)).dropna()
            return float(returns.std())
        return 0.001

    def _extract_rolling_std(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        if idx >= 20:
            return float(df.iloc[idx-20:idx+1]["Close"].std())
        return 0.001

    def _extract_hour(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> int:
        dt = pd.to_datetime(df.iloc[idx]["Datetime"])
        return int(dt.hour)

    def _extract_weekday(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> int:
        dt = pd.to_datetime(df.iloc[idx]["Datetime"])
        return int(dt.weekday())

    def _extract_session(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> str:
        # Simple timezone/session extraction (e.g. Asia/London/New York)
        dt = pd.to_datetime(df.iloc[idx]["Datetime"])
        h = dt.hour
        if 0 <= h < 8:
            return "Asian"
        elif 8 <= h < 13:
            return "London"
        elif 13 <= h < 17:
            return "London/NY"
        elif 17 <= h < 22:
            return "NewYork"
        else:
            return "Asian"

    def _extract_trend_score(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        # Linear regression r2 slope confirmation over last 20 bars
        if idx >= 20:
            y = df.iloc[idx-20:idx+1]["Close"].to_numpy()
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
            return float(r2 if not np.isnan(r2) else 0.0)
        return 0.0

    def _extract_range_score(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        # Ratio of high-low range over the period to total path traveled
        if idx >= 20:
            window = df.iloc[idx-20:idx+1]
            high_low = window["High"].max() - window["Low"].min()
            path = abs(window["Close"] - window["Close"].shift(1)).sum()
            return float(high_low / (path + 1e-9))
        return 0.5

    def _extract_compression_score(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        if idx >= 5:
            current_range = df.iloc[idx]["High"] - df.iloc[idx]["Low"]
            avg_past_range = (df.iloc[idx-5:idx]["High"] - df.iloc[idx-5:idx]["Low"]).mean()
            return float(current_range / (avg_past_range + 1e-9))
        return 1.0

    def _extract_distance_to_nearest_high(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        close = row["Close"]
        atr = row.get("atr_14", 0.0001)
        highs = [s.price for s in msg.swing_highs if s.confirmation_candle <= idx]
        if highs:
            nearest = min(highs, key=lambda h: abs(h - close))
            return float(abs(nearest - close) / atr)
        return 1.0

    def _extract_distance_to_nearest_low(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        close = row["Close"]
        atr = row.get("atr_14", 0.0001)
        lows = [s.price for s in msg.swing_lows if s.confirmation_candle <= idx]
        if lows:
            nearest = min(lows, key=lambda l: abs(l - close))
            return float(abs(nearest - close) / atr)
        return 1.0

    def _extract_distance_to_structure_break(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        close = row["Close"]
        atr = row.get("atr_14", 0.0001)
        breaks = [b.broken_level for b in msg.bos if b.index <= idx]
        if breaks:
            last_break = breaks[-1]
            return float(abs(last_break - close) / atr)
        return 1.0

    def _extract_distance_to_invalidation_level(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        # Use nearest low (for buy) or nearest high (for sell) as invalidation level proxy
        row = df.iloc[idx]
        close = row["Close"]
        atr = row.get("atr_14", 0.0001)
        lows = [s.price for s in msg.swing_lows if s.confirmation_candle <= idx]
        if lows:
            last_low = lows[-1]
            return float(abs(close - last_low) / atr)
        return 1.5

    def _extract_risk_reward_estimate(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        from Trade_Execution.location_engine import TradeLocationEngine
        tle = TradeLocationEngine()
        row = df.iloc[idx]
        close = row["Close"]
        # Use overall trend to decide direction
        direction = 1 if row.get("trend", 0) >= 0 else -1
        try:
            levels = tle.get_trade_levels(msg, direction, close)
            return float(levels.get("rr_ratio", 2.0))
        except Exception:
            return 2.0

    def _extract_ema50_distance_v1(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "ema_50" in row:
            return float(row["Close"] - row["ema_50"])
        return 0.0

    def _extract_ema50_distance_v2(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> float:
        row = df.iloc[idx]
        if "ema_50" in row and "atr_14" in row:
            return float((row["Close"] - row["ema_50"]) / (row["atr_14"] + 1e-9))
        return 0.0
