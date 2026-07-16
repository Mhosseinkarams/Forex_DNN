import os
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union

from Configs.path_manager import PathManager
from ML.decision_context import DecisionContext
from Collecting_Data.position_lifecycle import PositionLifecycle

logger = logging.getLogger("TradeFeatureRecorder")

class TradeFeatureRecorder:
    """
    Purpose:
        Record every signal candidate and every executed trade for future ML retraining.
        This provides a standardized, tabular, fully-flattened format directly consumable
        by future training pipelines.

    Architecture:
        Thread-safe daily rolling storage (CSV, Parquet, and future database backends).
        Supports live trading, simulation, notebooks, and backtesting.
    """
    def __init__(
        self,
        storage_dir: str = None,
        file_format: str = "csv",  # "csv", "parquet", or "both"
        compression: Optional[str] = None,
    ):
        if storage_dir is None:
            storage_dir = PathManager.get_relative_path("feature_data")
        self.storage_dir = storage_dir
        self.file_format = file_format.lower()
        self.compression = compression
        self._lock = threading.Lock()

        # Ensure directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        logger.info(f"TradeFeatureRecorder initialized in directory: {self.storage_dir} (Format: {self.file_format})")

    def _get_daily_filepath(self, date_str: str, file_ext: str) -> str:
        """Resolve the daily rolling filename."""
        # Sanitized date representation YYYYMMDD
        sanitized_date = date_str.replace("-", "").replace("/", "")[:8]
        return os.path.join(self.storage_dir, f"recorded_features_{sanitized_date}.{file_ext}")

    def record_candidate(
        self,
        signal_id: str,
        timestamp: str,
        strategy: str,
        symbol: str,
        timeframe: str,
        direction: str,  # "BUY" or "SELL"
        features: Dict[str, Any],
        decision_context: Optional[DecisionContext] = None,
        accepted: bool = True,
        reason: str = ""
    ) -> None:
        """
        Record a candidate signal event, creating a new row in the daily rolling file.
        All trade outcome fields are initialized to default placeholder values.
        """
        with self._lock:
            try:
                # 1. Gather metadata and candidate-level fields
                row_data = {
                    "signal_id": str(signal_id),
                    "timestamp": str(timestamp),
                    "strategy": str(strategy),
                    "symbol": str(symbol),
                    "timeframe": str(timeframe),
                    "direction": str(direction),
                    "accepted": int(accepted),
                    "rejected": int(not accepted),
                    "reason": str(reason),
                }

                # 2. Extract ML predictions and policy fields from DecisionContext
                if decision_context:
                    row_data.update({
                        "predicted_state": str(decision_context.predicted_state),
                        "state_confidence": float(decision_context.state_confidence),
                        "break_probability": float(decision_context.break_probability),
                        "rejection_probability": float(decision_context.rejection_probability),
                        "trade_quality_score": float(decision_context.trade_quality_score),
                        "confidence_score": float(decision_context.confidence_score),
                        "policy_allow_trade": int(decision_context.policy_recommendation.allow_trade),
                        "policy_risk_multiplier": float(decision_context.policy_recommendation.suggested_risk_multiplier),
                        "policy_position_scale": float(decision_context.policy_recommendation.suggested_position_scale),
                        "policy_tp_mode": str(decision_context.policy_recommendation.suggested_tp_mode),
                        "policy_sl_adjustment": float(decision_context.policy_recommendation.suggested_sl_adjustment),
                    })
                else:
                    row_data.update({
                        "predicted_state": "",
                        "state_confidence": np.nan,
                        "break_probability": np.nan,
                        "rejection_probability": np.nan,
                        "trade_quality_score": np.nan,
                        "confidence_score": np.nan,
                        "policy_allow_trade": 0,
                        "policy_risk_multiplier": 1.0,
                        "policy_position_scale": 1.0,
                        "policy_tp_mode": "",
                        "policy_sl_adjustment": 0.0,
                    })

                # 3. Incorporate all extracted features
                for k, v in features.items():
                    row_data[f"feature_{k}"] = v

                # 4. Initialize outcome fields with defaults
                outcome_placeholders = {
                    "trade_entry_price": np.nan,
                    "trade_exit_price": np.nan,
                    "trade_duration": np.nan,
                    "trade_sl": np.nan,
                    "trade_tp": np.nan,
                    "trade_rr": np.nan,
                    "trade_profit": np.nan,
                    "trade_drawdown": np.nan,
                    "trade_outcome": "",
                    "trade_manual_exit": 0,
                    "trade_tp1": np.nan,
                    "trade_tp2": np.nan,
                    "trade_stop_loss": np.nan,
                    "trade_trailing_stop": np.nan,
                    "trade_exit_reason": "",
                }
                row_data.update(outcome_placeholders)

                # 5. Write to Daily Rolling files
                self._save_row_to_files(row_data, timestamp)
                logger.info(f"Recorded signal candidate: {signal_id} ({symbol} {timeframe})")
            except Exception as e:
                logger.error(f"Failed to record signal candidate {signal_id}: {e}", exc_info=True)

    def record_outcome(self, signal_id: str, lifecycle: PositionLifecycle) -> None:
        """
        Locate the signal candidate row by signal_id in the daily rolling storage,
        and append all completed trade outcome parameters.
        """
        with self._lock:
            try:
                # 1. Extract outcome metrics from PositionLifecycle
                outcome = lifecycle.outcome
                execution = lifecycle.execution
                management = lifecycle.management

                is_manual = 1 if "manual" in str(outcome.strategy_reason).lower() else 0

                tp_prices = {}
                # Extract TP prices from management events if available, or fallback
                # Since we don't have exit manager state here, we can infer SL and TP details
                tp1 = execution.initial_take_profit if outcome.strategy_reason == "tp1" else np.nan
                tp2 = execution.initial_take_profit if outcome.strategy_reason == "tp2" else np.nan

                # Compute risk-reward ratio
                rr = outcome.r_multiple
                if pd.isna(rr) or rr == 0.0:
                    try:
                        risk_dist = abs(execution.actual_entry - execution.initial_stop_loss)
                        reward_dist = abs(execution.initial_take_profit - execution.actual_entry)
                        rr = reward_dist / (risk_dist + 1e-9)
                    except:
                        rr = np.nan

                outcome_updates = {
                    "trade_entry_price": float(execution.actual_entry),
                    "trade_exit_price": float(outcome.close_price),
                    "trade_duration": float(outcome.duration),
                    "trade_sl": float(execution.initial_stop_loss),
                    "trade_tp": float(execution.initial_take_profit),
                    "trade_rr": float(rr),
                    "trade_profit": float(outcome.realized_profit),
                    "trade_drawdown": float(management.maximum_adverse_excursion),
                    "trade_outcome": str(outcome.result),
                    "trade_manual_exit": int(is_manual),
                    "trade_tp1": float(tp1),
                    "trade_tp2": float(tp2),
                    "trade_stop_loss": float(outcome.close_price if outcome.result == "LOSS" else execution.initial_stop_loss),
                    "trade_trailing_stop": float(management.trailing_events[-1].get("new_sl", np.nan) if management.trailing_events else np.nan),
                    "trade_exit_reason": str(outcome.strategy_reason),
                }

                # 2. Search for the signal across all daily files in the directory
                updated = self._update_row_in_files(str(signal_id), outcome_updates)
                if updated:
                    logger.info(f"Recorded trade outcome for signal_id: {signal_id} (Outcome: {outcome.result})")
                else:
                    logger.warning(f"Could not find signal_id {signal_id} in recorded features files to append outcome.")

            except Exception as e:
                logger.error(f"Failed to record trade outcome for signal {signal_id}: {e}", exc_info=True)

    def _save_row_to_files(self, row_data: Dict[str, Any], timestamp_str: str) -> None:
        """Write a new row of data to the rolling daily files."""
        # Use ISO timestamp to extract date
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y%m%d")
        except Exception:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

        df_new = pd.DataFrame([row_data])

        # Save as CSV
        if self.file_format in ["csv", "both"]:
            filepath = self._get_daily_filepath(date_str, "csv")
            if os.path.exists(filepath):
                try:
                    df_existing = pd.read_csv(filepath)
                    # Align columns
                    combined = pd.concat([df_existing, df_new], ignore_index=True)
                    combined.to_csv(filepath, index=False)
                except Exception as e:
                    logger.error(f"Error appending CSV to {filepath}: {e}")
                    df_new.to_csv(filepath, index=False)
            else:
                df_new.to_csv(filepath, index=False)

        # Save as Parquet
        if self.file_format in ["parquet", "both"]:
            filepath = self._get_daily_filepath(date_str, "parquet")
            if os.path.exists(filepath):
                try:
                    df_existing = pd.read_parquet(filepath)
                    combined = pd.concat([df_existing, df_new], ignore_index=True)
                    combined.to_parquet(filepath, index=False, compression=self.compression)
                except Exception as e:
                    logger.error(f"Error appending Parquet to {filepath}: {e}")
                    df_new.to_parquet(filepath, index=False, compression=self.compression)
            else:
                df_new.to_parquet(filepath, index=False, compression=self.compression)

    def _update_row_in_files(self, signal_id: str, updates: Dict[str, Any]) -> bool:
        """Find and update a row with updates dictionary by matching signal_id."""
        files = os.listdir(self.storage_dir)
        found = False

        # Process CSV files
        if self.file_format in ["csv", "both"]:
            csv_files = [f for f in files if f.endswith(".csv")]
            for f in csv_files:
                filepath = os.path.join(self.storage_dir, f)
                try:
                    df = pd.read_csv(filepath)
                    # Force signal_id to be compared as string
                    df["signal_id"] = df["signal_id"].astype(str)
                    if "signal_id" in df.columns and signal_id in df["signal_id"].values:
                        idx = df[df["signal_id"] == signal_id].index[0]
                        for k, v in updates.items():
                            if k in df.columns:
                                # If updating with a string, coerce the column to object first
                                if isinstance(v, str) and df[k].dtype != object:
                                    df[k] = df[k].astype(object)
                            df.at[idx, k] = v
                        df.to_csv(filepath, index=False)
                        found = True
                except Exception as e:
                    logger.error(f"Error updating CSV file {filepath}: {e}")

        # Process Parquet files
        if self.file_format in ["parquet", "both"]:
            parquet_files = [f for f in files if f.endswith(".parquet")]
            for f in parquet_files:
                filepath = os.path.join(self.storage_dir, f)
                try:
                    df = pd.read_parquet(filepath)
                    df["signal_id"] = df["signal_id"].astype(str)
                    if "signal_id" in df.columns and signal_id in df["signal_id"].values:
                        idx = df[df["signal_id"] == signal_id].index[0]
                        for k, v in updates.items():
                            if k in df.columns:
                                if isinstance(v, str) and df[k].dtype != object:
                                    df[k] = df[k].astype(object)
                            df.at[idx, k] = v
                        df.to_parquet(filepath, index=False, compression=self.compression)
                        found = True
                except Exception as e:
                    logger.error(f"Error updating Parquet file {filepath}: {e}")

        return found
