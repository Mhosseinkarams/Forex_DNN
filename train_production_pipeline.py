#!/usr/bin/env python3
"""
train_production_pipeline.py

A production-grade training pipeline for the Forex_DNN framework.
Capable of processing historical datasets, building a consolidated ML dataset,
executing thorough diagnostics, training models, and generating detailed reports.

Usage examples:
    # Train Market State Classifier on all available symbols
    python train_production_pipeline.py --model-name MarketStateClassifier --timeframe M5

    # Train Level Break Probability Model on specific symbols with custom window size
    python train_production_pipeline.py --model-name LevelBreakProbabilityModel --symbol EURUSD,GBPUSD --window-size 35
"""

import os
import re
import sys
import argparse
import logging
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd

# Metric imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

# Plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTS_SUPPORTED = True
except ImportError:
    PLOTS_SUPPORTED = False

# SHAP
try:
    import shap
    SHAP_SUPPORTED = True
except ImportError:
    SHAP_SUPPORTED = False

# Forex_DNN imports
from ML.feature_registry import FeatureRegistry
from ML.feature_pipeline import FeaturePipeline
from ML.data_cleaner import DataCleaner
from ML.dataset_validator import DatasetValidator
from ML.label_engine import LabelEngine
from ML.market_state_labeler import MarketStateLabeler
from ML.dataset_builder import DatasetBuilder
from ML.models.market_state_classifier import MarketStateClassifier
from ML.models.level_break_probability import LevelBreakProbabilityModel
from ML.trainer import Trainer
from ML.evaluator import Evaluator

from Market_Data_Pipeline.historical_dataset_builder import HistoricalDatasetBuilder
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.structure_graph import MarketStructureGraph


def setup_logging(experiment_dir: str) -> logging.Logger:
    """Configures logging to output to both console and training.log file."""
    os.makedirs(experiment_dir, exist_ok=True)
    log_file = os.path.join(experiment_dir, "training.log")

    logger = logging.getLogger("ProductionPipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False # Avoid duplicate logging in training.log via propagation to root
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Redirect root logging as well to capture sub-engine logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    return logger


def discover_files(data_dir: str, timeframe: str, symbol_filter: str = "all") -> dict:
    """
    Discovers Parquet and CSV files for the given timeframe.
    Supports flat structure (SYMBOL_M5.parquet) and nested structure (SYMBOL/M5.parquet).
    """
    discovered = {}
    if not os.path.exists(data_dir):
        return discovered

    # 1. Check nested directory structure (e.g. HistoricalData/EURUSD/M5.parquet)
    for item in os.listdir(data_dir):
        subdir = os.path.join(data_dir, item)
        if os.path.isdir(subdir):
            # Try Parquet
            parquet_path = os.path.join(subdir, f"{timeframe}.parquet")
            if os.path.exists(parquet_path):
                discovered[item] = parquet_path
                continue
            # Try CSV
            csv_path = os.path.join(subdir, f"{timeframe}.csv")
            if os.path.exists(csv_path):
                discovered[item] = csv_path
                continue

    # 2. Check flat directory structure (e.g. HistoricalData/EURUSD_M5.parquet)
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            match = re.match(r"^([A-Z0-9]+)_" + re.escape(timeframe) + r"\.(parquet|csv)$", file, re.IGNORECASE)
            if match:
                sym = match.group(1)
                # If we already got it nested, skip
                if sym not in discovered:
                    discovered[sym] = file_path

    # Apply symbol filter
    if symbol_filter != "all":
        target_symbols = [s.strip() for s in symbol_filter.split(",")]
        filtered = {}
        for sym in target_symbols:
            # Try exact match or standardizing
            matched = False
            for disc_sym, path in discovered.items():
                if disc_sym.upper() == sym.upper():
                    filtered[disc_sym] = path
                    matched = True
            if not matched:
                print(f"Warning: Requested symbol '{sym}' was not discovered in {data_dir}.")
        discovered = filtered

    return discovered


def load_raw_file(file_path: str) -> pd.DataFrame:
    """Loads Parquet or CSV historical candle file."""
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format for: {file_path}")

    # Standardize columns
    for col in df.columns:
        if col.lower() == "datetime":
            df.rename(columns={col: "Datetime"}, inplace=True)
            break

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.sort_values(by="Datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def calculate_per_symbol_stats(
    symbol: str,
    timeframe: str,
    df_raw: pd.DataFrame,
    df_labeled: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    ms_engine: MarketStructureEngine,
    sd_engine: SupplyDemandEngine,
    unlabeled_count: int,
    discarded_count: int
) -> dict:
    """Calculates granular statistics about market structure, S/D zones, and features."""
    date_range = ["N/A", "N/A"]
    if "Datetime" in df_raw.columns and not df_raw.empty:
        date_range = [str(df_raw["Datetime"].min()), str(df_raw["Datetime"].max())]

    # Label distribution
    label_dist = {}
    if "target" in df_cleaned.columns:
        counts = df_cleaned["target"].value_counts().to_dict()
        label_dist = {str(k): int(v) for k, v in counts.items()}

    # Market structure counts
    bull_bos = sum(1 for b in ms_engine.bos_list if b.direction == 1)
    bear_bos = sum(1 for b in ms_engine.bos_list if b.direction == -1)
    bull_choch = sum(1 for c in ms_engine.choch_list if c.new_trend == 1)
    bear_choch = sum(1 for c in ms_engine.choch_list if c.new_trend == -1)

    # S/D zone counts
    supply_zones = [z for z in sd_engine.zones if z.type == "Supply"]
    demand_zones = [z for z in sd_engine.zones if z.type == "Demand"]

    avg_strength = 0.0
    if sd_engine.zones:
        avg_strength = float(np.mean([z.strength_score for z in sd_engine.zones]))

    broken_zones = sum(1 for z in sd_engine.zones if z.broken)
    fresh_zones = sum(1 for z in sd_engine.zones if z.freshness)

    # Feature quality metrics
    nan_count = int(df_labeled.isna().sum().sum())
    duplicate_count = int(df_labeled.duplicated().sum())

    constant_features = []
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["target", "confidence", "window_start", "window_end", "Open", "High", "Low", "Close", "TickVolume", "ema_50", "ema_600", "ema_800"]:
            if df_cleaned[col].nunique() <= 1:
                constant_features.append(col)

    stats = {
        "symbol": symbol,
        "timeframe": timeframe,
        "date_range": date_range,
        "candles_count": len(df_raw),
        "generated_samples": len(df_labeled),
        "discarded_samples": discarded_count,
        "unlabeled_samples": unlabeled_count,
        "final_dataset_size": len(df_cleaned),
        "label_distribution": label_dist,
        "market_structure": {
            "bull_bos": bull_bos,
            "bear_bos": bear_bos,
            "bull_choch": bull_choch,
            "bear_choch": bear_choch
        },
        "supply_demand": {
            "supply_zones": len(supply_zones),
            "demand_zones": len(demand_zones),
            "average_zone_strength": avg_strength,
            "broken_zones": broken_zones,
            "fresh_zones": fresh_zones
        },
        "feature_quality": {
            "nans": nan_count,
            "constant_features_count": len(constant_features),
            "constant_features_list": constant_features,
            "duplicate_rows": duplicate_count
        }
    }
    return stats


def generate_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list,
    model,
    feature_cols: list,
    X_test_global: np.ndarray,
    output_dir: str
):
    """Generates and saves the five standard diagnostic performance figures."""
    if not PLOTS_SUPPORTED:
        return

    # 1. Confusion Matrix Plot
    try:
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix Heatmap")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

    # Binary vs Multiclass Plotting
    is_binary = len(classes) == 2

    # 2. ROC Curve Plot
    try:
        plt.figure(figsize=(7, 6))
        if is_binary:
            if y_proba.ndim == 2:
                p_scores = y_proba[:, 1]
            else:
                p_scores = y_proba
            fpr, tpr, _ = roc_curve(y_test, p_scores)
            auc_val = roc_auc_score(y_test, p_scores)
            plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.4f})", color="darkorange", lw=2)
        else:
            # Plot micro-average ROC or per-class ROC
            for i, class_name in enumerate(classes):
                if i >= y_proba.shape[1]:
                    continue
                # Binarize labels
                y_test_bin = (y_test == i).astype(int)
                fpr, tpr, _ = roc_curve(y_test_bin, y_proba[:, i])
                auc_val = roc_auc_score(y_test_bin, y_proba[:, i])
                plt.plot(fpr, tpr, label=f"Class {class_name} (AUC = {auc_val:.4f})", lw=1.5)

        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")

    # 3. Precision-Recall Curve Plot
    try:
        plt.figure(figsize=(7, 6))
        if is_binary:
            if y_proba.ndim == 2:
                p_scores = y_proba[:, 1]
            else:
                p_scores = y_proba
            prec, rec, _ = precision_recall_curve(y_test, p_scores)
            plt.plot(rec, prec, label="Precision-Recall Curve", color="purple", lw=2)
        else:
            for i, class_name in enumerate(classes):
                if i >= y_proba.shape[1]:
                    continue
                y_test_bin = (y_test == i).astype(int)
                prec, rec, _ = precision_recall_curve(y_test_bin, y_proba[:, i])
                plt.plot(rec, prec, label=f"Class {class_name}", lw=1.5)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "precision_recall.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting Precision-Recall curve: {e}")

    # 4. Calibration Curve Plot
    try:
        plt.figure(figsize=(7, 6))
        if is_binary:
            if y_proba.ndim == 2:
                p_scores = y_proba[:, 1]
            else:
                p_scores = y_proba
            prob_true, prob_pred = calibration_curve(y_test, p_scores, n_bins=10)
            plt.plot(prob_pred, prob_true, marker="o", linewidth=1.5, label="Calibrated Model")
        else:
            for i, class_name in enumerate(classes):
                if i >= y_proba.shape[1]:
                    continue
                y_test_bin = (y_test == i).astype(int)
                prob_true, prob_pred = calibration_curve(y_test_bin, y_proba[:, i], n_bins=10)
                plt.plot(prob_pred, prob_true, marker="s", label=f"Class {class_name}")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Probability Calibration Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting calibration curve: {e}")

    # 5. SHAP Summary Plot
    if SHAP_SUPPORTED:
        try:
            # We sample at most 300 samples for swift calculation
            sample_size = min(300, len(y_test))
            # Fit tree explainer
            # Convert X_test to DataFrame matching trained features
            # Get model engine
            eng = model.model
            explainer = shap.TreeExplainer(eng)
            # Create sample data
            X_sample = pd.DataFrame(X_test_global[:sample_size], columns=feature_cols) if X_test_global is not None else None
            if X_sample is not None:
                shap_values = explainer.shap_values(X_sample)

                plt.figure(figsize=(8, 6))
                # For multiclass, shap_values might be a list of arrays (one per class)
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                else:
                    shap.summary_plot(shap_values, X_sample, show=False)
                plt.title("SHAP Feature Importance Summary")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150)
                plt.close()
        except Exception as e:
            print(f"Error plotting SHAP summary: {e}")
            # Create a placeholder empty plot to comply with output requirements
            create_empty_placeholder(output_dir, "shap_summary.png")
    else:
        # Create placeholder
        create_empty_placeholder(output_dir, "shap_summary.png")


def create_empty_placeholder(output_dir: str, filename: str):
    """Creates a basic placeholder plot when plotting libraries or SHAP are missing."""
    if not PLOTS_SUPPORTED:
        return
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, f"Plot '{filename}' skipped\n(Library not installed or computation skipped)",
             ha="center", va="center", fontsize=11, color="gray")
    plt.title(filename)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=100)
    plt.close()


def run_pipeline():
    parser = argparse.ArgumentParser(description="Forex_DNN Production Training Pipeline")
    parser.add_argument("--model-name", type=str, default="MarketStateClassifier",
                        choices=["MarketStateClassifier", "LevelBreakProbabilityModel"],
                        help="The type of model wrapper to build and train.")
    parser.add_argument("--model-type", type=str, default="lightgbm",
                        choices=["lightgbm", "randomforest"],
                        help="The underlying machine learning algorithm.")
    parser.add_argument("--data-dir", type=str, default="HistoricalData",
                        help="Input historical data folder.")
    parser.add_argument("--symbol", type=str, default="all",
                        help="Specific symbol(s) list (comma-separated, e.g. EURUSD,GBPUSD) or 'all'.")
    parser.add_argument("--timeframe", type=str, default="M5",
                        help="Timeframe of files to discover (e.g. M5, M15).")
    parser.add_argument("--window-size", type=int, default=35,
                        help="Window size for sliding iterations.")
    parser.add_argument("--window-stride", type=int, default=1,
                        help="Stride for sliding iterations.")
    parser.add_argument("--max-bars", type=int, default=0,
                        help="Use only the most recent N bars per symbol (0 keeps all bars).")
    parser.add_argument("--cache-dir", type=str, default="cache",
                        help="Folder containing cached processed symbol datasets.")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--force", action="store_true",
                        help="Force training even if critical diagnostics warnings exist.")
    parser.add_argument("--register-production", action="store_true",
                        help="Register the trained model as production after evaluation. Disabled by default.")

    args = parser.parse_args()

    # Generate unique timestamped experiment folder
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, "experiments", exp_id)
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize Logger
    logger = setup_logging(experiment_dir)

    logger.info("==========================================================")
    logger.info(f"    Forex_DNN Production ML Pipeline — Run: {exp_id}")
    logger.info("==========================================================")
    logger.info(f"Target Model       : {args.model_name}")
    logger.info(f"Learning Backend   : {args.model_type.upper()}")
    logger.info(f"Data Directory     : {args.data_dir}")
    logger.info(f"Timeframe          : {args.timeframe}")
    logger.info(f"Window / Stride    : {args.window_size} / {args.window_stride}")
    logger.info(f"Experiment Folder  : {experiment_dir}")

    # 1. Discover data files
    logger.info("Discovering historical data files...")
    discovered_files = discover_files(args.data_dir, args.timeframe, args.symbol)
    if not discovered_files:
        logger.error(f"No files found in '{args.data_dir}' matching timeframe '{args.timeframe}' and filter '{args.symbol}'.")
        sys.exit(1)

    logger.info(f"Successfully discovered {len(discovered_files)} symbols: {list(discovered_files.keys())}")

    # 2. Process Symbols individually (Memory-Efficient & Cacheable)
    master_dfs = []
    per_symbol_stats_map = {}

    # Setup core engines and registry for processing
    registry = FeatureRegistry(load_defaults=True)
    ms_engine = MarketStructureEngine(lookback=3)
    sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
    cleaner = DataCleaner()

    # Label coordinator for Market State
    label_engine_coordinator = LabelEngine(
        window_size=args.window_size,
        window_stride=args.window_stride,
        registry=registry
    )

    # Level break dataset builder if level break requested
    level_break_builder = DatasetBuilder(registry=registry)

    for sym, filepath in discovered_files.items():
        logger.info(f"--- Processing Symbol: {sym} (Path: {filepath}) ---")

        # Create localized cache path
        # Cache unique key matches symbol, timeframe, window configs, and model target
        # Bump this whenever feature/label causality changes.  Old caches can
        # otherwise preserve features computed with a later version of the
        # structural pipeline.
        cache_filename = f"causal_v3_{sym}_{args.timeframe}_w{args.window_size}_s{args.window_stride}_{args.model_name}.parquet"
        cache_path = os.path.join(args.cache_dir, cache_filename)

        df_sym_cleaned = None
        df_sym_raw = load_raw_file(filepath)
        if args.max_bars > 0 and len(df_sym_raw) > args.max_bars:
            df_sym_raw = df_sym_raw.tail(args.max_bars).reset_index(drop=True)
            logger.info(f"Limited {sym} to its most recent {len(df_sym_raw)} bars.")
        df_sym_labeled = None

        unlabeled_count = 0
        discarded_count = 0

        # Attempt Cache Load
        if os.path.exists(cache_path):
            try:
                logger.info(f"Loading pre-computed cached dataset from {cache_path}")
                df_sym_cleaned = pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to read cached file: {e}. Re-computing...")

        if df_sym_cleaned is None:
            logger.info("Computing sliding-window features, structure metrics, and targets...")

            # Run sequential indicators
            from Collecting_Data.indicators import IndicatorEngine
            ind_engine = IndicatorEngine(ema_periods=[50, 600, 800], slope_period=32)
            df_enriched = ind_engine.calculate(df_sym_raw)

            # Run structure and supply/demand
            df_enriched = ms_engine.process(df_enriched)
            df_enriched = sd_engine.process(df_enriched)

            # Construct graph
            msg = MarketStructureGraph(
                symbol=sym,
                timeframe=args.timeframe,
                swing_highs=[s for s in ms_engine.swings if s.level_type == "SwingHigh"],
                swing_lows=[s for s in ms_engine.swings if s.level_type == "SwingLow"],
                protected_high=ms_engine.protected_high,
                protected_low=ms_engine.protected_low,
                bos=list(ms_engine.bos_list),
                choch=list(ms_engine.choch_list),
                supply_zones=[z for z in sd_engine.zones if z.type == "Supply"],
                demand_zones=[z for z in sd_engine.zones if z.type == "Demand"],
                trend_direction="Bull" if df_enriched.iloc[-1].get("trend", 0) == 1 else ("Bear" if df_enriched.iloc[-1].get("trend", 0) == -1 else "Neutral"),
                atr=float(df_enriched.iloc[-1].get("atr_14", 0.0001))
            )

            # Generate target labels based on requested model
            if args.model_name == "MarketStateClassifier":
                df_sym_labeled = label_engine_coordinator.generate_dataset(
                    data_inputs={(sym, args.timeframe): df_sym_raw},
                    ms_engine=ms_engine,
                    sd_engine=sd_engine
                )
                # Recover unlabeled (discarded) counts from label engine coordinator
                unlabeled_count = label_engine_coordinator.removed_samples_count
                target_col_name = "target"
            else:
                # Level Break Probability Model
                df_sym_labeled = level_break_builder.build_level_break_dataset(df_enriched, msg)
                # Compute unlabeled count as total possible approaches minus generated approaches
                # (For level break, we only log when price approaches a zone, so skipped rows are normal)
                unlabeled_count = len(df_enriched) - len(df_sym_labeled)
                target_col_name = "target"

            # Apply DataCleaner
            if not df_sym_labeled.empty:
                # Keep original string representations to prevent DataCleaner from encoding them to category codes
                orig_symbol = df_sym_labeled["symbol"].copy() if "symbol" in df_sym_labeled.columns else None
                orig_timeframe = df_sym_labeled["timeframe"].copy() if "timeframe" in df_sym_labeled.columns else None
                orig_datetime = df_sym_labeled["datetime"].copy() if "datetime" in df_sym_labeled.columns else None

                df_sym_cleaned = cleaner.clean(df_sym_labeled, label_col=target_col_name)

                # Restore original string values so that they are correct in final reports
                if orig_symbol is not None:
                    df_sym_cleaned["symbol"] = orig_symbol
                if orig_timeframe is not None:
                    df_sym_cleaned["timeframe"] = orig_timeframe
                if orig_datetime is not None:
                    # Keep real event time for the global chronological split;
                    # DataCleaner may numerically encode object columns.
                    df_sym_cleaned["datetime"] = orig_datetime

                discarded_count = len(df_sym_labeled) - len(df_sym_cleaned)
            else:
                df_sym_cleaned = pd.DataFrame()

            # Cache the cleaned dataset for fast resume support
            if not df_sym_cleaned.empty:
                os.makedirs(args.cache_dir, exist_ok=True)
                df_sym_cleaned.to_parquet(cache_path, index=False)
                logger.info(f"Cached processed symbol dataset to: {cache_path}")

        else:
            # When loading from cache, run fast engine process to extract correct counts for stats
            logger.info("Executing fast structural engines for stats report generation...")
            # Enforce calculations for stats map
            df_enriched = ms_engine.process(df_sym_raw)
            df_enriched = sd_engine.process(df_enriched)
            # Create a mock labeled df if cache loaded
            df_sym_labeled = df_sym_cleaned

        if not df_sym_cleaned.empty:
            master_dfs.append(df_sym_cleaned)

            # Compute symbol-level stats
            sym_stats = calculate_per_symbol_stats(
                sym, args.timeframe, df_sym_raw, df_sym_labeled, df_sym_cleaned,
                ms_engine, sd_engine, unlabeled_count, discarded_count
            )
            per_symbol_stats_map[sym] = sym_stats

            # Save individual symbol JSON report in experiment folder
            sym_json_path = os.path.join(experiment_dir, f"{sym}_statistics.json")
            with open(sym_json_path, "w") as f:
                json.dump(sym_stats, f, indent=4)
            logger.info(f"Saved stats report for {sym} to {sym_json_path}")
        else:
            logger.warning(f"No samples generated for symbol {sym}.")

    if not master_dfs:
        logger.error("No training samples were generated across any symbols. Aborting.")
        sys.exit(1)

    # 3. Consolidate Master Dataset
    logger.info("Consolidating all processed symbol datasets into Master Dataset...")
    master_df = pd.concat(master_dfs, ignore_index=True)

    # Enforce strict chronological ordering globally
    if "datetime" in master_df.columns:
        master_df["_parsed_dt"] = pd.to_datetime(master_df["datetime"])
        master_df.sort_values(by=["_parsed_dt", "symbol"], inplace=True)
        master_df.drop(columns=["_parsed_dt"], inplace=True)
        master_df.reset_index(drop=True, inplace=True)

    # 4. Global Dataset Statistics Analysis
    total_symbols = master_df["symbol"].nunique() if "symbol" in master_df.columns else 0
    total_candles = sum(stats["candles_count"] for stats in per_symbol_stats_map.values())
    total_samples = len(master_df)
    memory_mb = master_df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Exclude metadata and raw price/EMA columns to compute Feature Count
    metadata_cols = [
        "target", "confidence", "window_start", "window_end", "Open", "High", "Low", "Close",
        "TickVolume", "ema_50", "ema_600", "ema_800", "sample_id", "symbol", "timeframe", "datetime", "zone_type",
        "label_version", "engine_version"
    ]
    feature_cols = [c for c in master_df.columns if c not in metadata_cols and not c.startswith("meta_labeler_")]
    feature_count = len(feature_cols)

    # Label distribution
    label_dist = master_df["target"].value_counts().to_dict()
    label_dist = {str(k): int(v) for k, v in label_dist.items()}

    # Breakdowns
    samples_per_symbol = master_df["symbol"].value_counts().to_dict() if "symbol" in master_df.columns else {}

    # Samples per year
    samples_per_year = {}
    if "datetime" in master_df.columns:
        try:
            years = pd.to_datetime(master_df["datetime"]).dt.year
            samples_per_year = years.value_counts().to_dict()
            samples_per_year = {str(k): int(v) for k, v in samples_per_year.items()}
        except Exception:
            pass

    # Samples per session
    samples_per_session = {}
    if "hour" in master_df.columns:
        # Asian: 22-8, London: 9-12, NY: 17-21, Overlap: 13-16
        hours = master_df["hour"]
        sessions = np.where(
            hours.isin([13, 14, 15, 16]), "London/NY Overlap",
            np.where(hours.isin([9, 10, 11, 12]), "London",
                     np.where(hours.isin([17, 18, 19, 20, 21]), "New York", "Asian"))
        )
        sessions_series = pd.Series(sessions)
        samples_per_session = sessions_series.value_counts().to_dict()
        samples_per_session = {str(k): int(v) for k, v in samples_per_session.items()}

    # Samples per volatility regime
    samples_per_vol = {}
    if "atr_percentile" in master_df.columns:
        vol_regimes = np.where(
            master_df["atr_percentile"] > 0.7, "High Volatility",
            np.where(master_df["atr_percentile"] < 0.3, "Low Volatility", "Medium Volatility")
        )
        samples_per_vol = pd.Series(vol_regimes).value_counts().to_dict()
        samples_per_vol = {str(k): int(v) for k, v in samples_per_vol.items()}

    global_summary = {
        "total_symbols": total_symbols,
        "total_candles": total_candles,
        "total_samples": total_samples,
        "dataset_size_mb": float(memory_mb),
        "feature_count": feature_count,
        "label_distribution": label_dist,
        "samples_per_symbol": samples_per_symbol,
        "samples_per_year": samples_per_year,
        "samples_per_market_session": samples_per_session,
        "samples_per_volatility_regime": samples_per_vol
    }

    # Save Global Summary Report
    global_report_path = os.path.join(experiment_dir, "dataset_summary.json")
    with open(global_report_path, "w") as f:
        json.dump(global_summary, f, indent=4)

    # Save Consolidated Per-Symbol statistics JSON in experiment directory
    with open(os.path.join(experiment_dir, "per_symbol_statistics.json"), "w") as f:
        json.dump(per_symbol_stats_map, f, indent=4)

    # Print summary reports on the console
    logger.info("\n" + "=" * 60)
    logger.info("                 GLOBAL DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Symbols                   : {total_symbols}")
    logger.info(f"Total Candles                   : {total_candles}")
    logger.info(f"Total Combined Samples          : {total_samples}")
    logger.info(f"Consolidated Memory Footprint   : {memory_mb:.2f} MB")
    logger.info(f"Feature Vector Columns          : {feature_count}")
    logger.info(f"Target Labels Breakdown         : {label_dist}")
    logger.info(f"Samples per Symbol Distribution : {samples_per_symbol}")
    logger.info(f"Samples per Year Distribution   : {samples_per_year}")
    logger.info(f"Samples per Session             : {samples_per_session}")
    logger.info(f"Samples per Volatility Regime   : {samples_per_vol}")
    logger.info("=" * 60 + "\n")

    # Generate static Matplotlib distribution plots for global summary
    if PLOTS_SUPPORTED:
        try:
            # Save label distribution chart
            plt.figure(figsize=(6, 4))
            plt.bar(list(label_dist.keys()), list(label_dist.values()), color="teal")
            plt.title("Label Distribution")
            plt.ylabel("Sample Count")
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_dir, "label_distribution.png"), dpi=100)
            plt.close()

            # Save symbol distribution chart
            plt.figure(figsize=(6, 4))
            plt.bar(list(samples_per_symbol.keys()), list(samples_per_symbol.values()), color="indigo")
            plt.title("Samples per Symbol")
            plt.ylabel("Sample Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_dir, "symbol_distribution.png"), dpi=100)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to save distribution charts: {e}")

    # 5. Save Processed Datasets
    os.makedirs(os.path.join(args.output_dir, "datasets"), exist_ok=True)
    if args.model_name == "MarketStateClassifier":
        master_parquet_path = os.path.join(args.output_dir, "datasets", "market_state_full.parquet")
    else:
        master_parquet_path = os.path.join(args.output_dir, "datasets", "level_break_full.parquet")

    logger.info(f"Saving fully consolidated master dataset to: {master_parquet_path}")
    master_df.to_parquet(master_parquet_path, index=False)

    # 6. Dataset Diagnostics validation before training
    logger.info("Running complete dataset diagnostics using DatasetValidator...")
    validator = DatasetValidator()
    # Ensure correct expected window size check
    diag_report = validator.validate(master_df, expected_window_size=args.window_size)

    # Log warnings and errors
    logger.info(f"Diagnostics Audit status: {'PASS' if diag_report['is_valid'] else 'FAIL'}")
    for warn in diag_report["warnings"]:
        logger.warning(f"DIAGNOSTIC WARNING: {warn}")
    for err in diag_report["errors"]:
        logger.error(f"DIAGNOSTIC CRITICAL ERROR: {err}")

    if not diag_report["is_valid"]:
        if args.force:
            logger.warning("CRITICAL ERROR: Dataset diagnostics failed, but training is FORCED. Continuing...")
        else:
            logger.error("CRITICAL ERROR: Dataset diagnostics failed! Training process aborted to maintain safety.")
            logger.error("Check logs above. Pass --force to override if necessary.")
            sys.exit(1)

    # 7. Model Initialization & Setup
    logger.info("Initializing Machine Learning model wrapper...")
    config_path = f"configs/{'market_state' if args.model_name == 'MarketStateClassifier' else 'level_break'}.yaml"
    if not os.path.exists(config_path):
        config_path = None

    if args.model_name == "MarketStateClassifier":
        model = MarketStateClassifier(
            model_type=args.model_type,
            config_path=config_path,
            random_state=args.seed
        )
        classes = ["TREND", "RANGE", "TRANSITION"]
        target_col = "target"

        # Explicitly map target string values to integers for perfect alignment
        # TREND=0, RANGE=1, TRANSITION=2
        class_mapping = {"TREND": 0, "RANGE": 1, "TRANSITION": 2}
        if master_df[target_col].dtype == object or isinstance(master_df[target_col].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(master_df[target_col]):
            logger.info("Enforcing string class target mapping: TREND=0, RANGE=1, TRANSITION=2")
            master_df[target_col] = master_df[target_col].map(class_mapping)

    else:
        model = LevelBreakProbabilityModel(
            model_type=args.model_type,
            config_path=config_path,
            random_state=args.seed
        )
        classes = ["REJECT", "BREAK"]
        target_col = "target"

    # Save Config file for reproducibility
    config_data = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "timeframe": args.timeframe,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "seed": args.seed,
        "feature_count": len(feature_cols),
        "total_samples": len(master_df),
        "target_column": target_col,
        "classes": classes,
        "hyperparameters": model.hyperparameters,
        "feature_hash": registry.compute_hash()
    }
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config_data, f, indent=4)

    # 8. Chronological train/validation/test splits
    # Chronological splitting splits the combined timeline: 70% Train, 15% Val, 15% Test
    n_samples = len(master_df)
    train_idx = int(n_samples * 0.7)
    val_idx = int(n_samples * 0.85)

    df_train = master_df.iloc[:train_idx]
    df_val = master_df.iloc[train_idx:val_idx]
    df_test = master_df.iloc[val_idx:]

    logger.info(f"Chronological Splits: Train={len(df_train)} | Val={len(df_val)} | Test={len(df_test)}")

    X_train = df_train[feature_cols]
    y_train = df_train[target_col].to_numpy()

    X_val = df_val[feature_cols]
    y_val = df_val[target_col].to_numpy()

    X_test_global = df_test[feature_cols].to_numpy()
    y_test_global = df_test[target_col].to_numpy()

    # 9. Model Fitting
    logger.info("Fitting model on Training split...")
    model.fit(
        X_train,
        y_train,
        feature_names=feature_cols,
        dataset_version=exp_id,
        dataset_hash=diag_report["metrics"].get("dataset_hash", "unknown")
    )

    # Calibrate probability if requested or default (use Platt Scaling / sigmoid on Val set)
    logger.info("Calibrating model probabilities on Validation split...")
    model.calibrate(X_val, y_val, method="sigmoid")

    # 10. Global Evaluation
    logger.info("Evaluating fitted model globally on the Test split...")
    inference_engine = model.calibrated_model if model.calibrated_model is not None else model.model
    y_pred_global = inference_engine.predict(X_test_global)
    y_proba_global = inference_engine.predict_proba(X_test_global)

    # Calculate statistics
    global_acc = float(accuracy_score(y_test_global, y_pred_global))
    is_binary = len(classes) == 2

    if is_binary:
        global_prec = float(precision_score(y_test_global, y_pred_global, average="binary", zero_division=0))
        global_rec = float(recall_score(y_test_global, y_pred_global, average="binary", zero_division=0))
        global_f1 = float(f1_score(y_test_global, y_pred_global, average="binary", zero_division=0))
        # Handle 1D or 2D probas
        if y_proba_global.ndim == 2:
            global_auc = float(roc_auc_score(y_test_global, y_proba_global[:, 1]))
        else:
            global_auc = float(roc_auc_score(y_test_global, y_proba_global))
    else:
        global_prec = float(precision_score(y_test_global, y_pred_global, average="weighted", zero_division=0))
        global_rec = float(recall_score(y_test_global, y_pred_global, average="weighted", zero_division=0))
        global_f1 = float(f1_score(y_test_global, y_pred_global, average="weighted", zero_division=0))
        global_auc = 0.0

    global_cm = confusion_matrix(y_test_global, y_pred_global, labels=list(range(len(classes)))).tolist()
    global_report_txt = classification_report(y_test_global, y_pred_global, labels=list(range(len(classes))), target_names=classes, zero_division=0)

    # 11. Per-Symbol Evaluation (Cross-market generalization test)
    logger.info("Evaluating model per symbol on Test split...")
    per_symbol_eval_map = {}

    for sym in discovered_files.keys():
        # Get symbol index slice in Test dataframe
        sym_test_df = df_test[df_test["symbol"] == sym]
        if sym_test_df.empty:
            logger.warning(f"Symbol '{sym}' has no samples inside chronological Test Split horizon.")
            continue

        X_sym_test = sym_test_df[feature_cols].to_numpy()
        y_sym_test = sym_test_df[target_col].to_numpy()

        y_sym_pred = inference_engine.predict(X_sym_test)
        y_sym_proba = inference_engine.predict_proba(X_sym_test)

        sym_acc = float(accuracy_score(y_sym_test, y_sym_pred))

        if is_binary:
            sym_f1 = float(f1_score(y_sym_test, y_sym_pred, average="binary", zero_division=0))
            if y_sym_proba.ndim == 2:
                sym_auc = float(roc_auc_score(y_sym_test, y_sym_proba[:, 1]))
            else:
                sym_auc = float(roc_auc_score(y_sym_test, y_sym_proba))
        else:
            sym_f1 = float(f1_score(y_sym_test, y_sym_pred, average="weighted", zero_division=0))
            sym_auc = 0.0

        sym_cm = confusion_matrix(y_sym_test, y_sym_pred, labels=list(range(len(classes)))).tolist()

        per_symbol_eval_map[sym] = {
            "test_samples": len(sym_test_df),
            "accuracy": sym_acc,
            "f1_score": sym_f1,
            "roc_auc": sym_auc,
            "confusion_matrix": sym_cm
        }

        # Print per-symbol results nicely on console
        logger.info(f"Symbol: {sym:<8} | Samples: {len(sym_test_df):<5} | Accuracy: {sym_acc:.4f} | F1: {sym_f1:.4f}")

    # Consolidate Metrics output
    metrics_json = {
        "global": {
            "accuracy": global_acc,
            "precision": global_prec,
            "recall": global_rec,
            "f1_score": global_f1,
            "roc_auc": global_auc,
            "confusion_matrix": global_cm
        },
        "per_symbol": per_symbol_eval_map
    }

    # Save Metrics File
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=4)
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    # Save classification report as plain text
    txt_report_path = os.path.join(experiment_dir, "classification_report.txt")
    with open(txt_report_path, "w") as f:
        f.write(global_report_txt)
    logger.info(f"Saved classification report to {txt_report_path}")

    # 12. Save Feature Importance CSV
    importance_map = model.get_feature_importance()
    sorted_importance = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)

    importance_path = os.path.join(experiment_dir, "feature_importance.csv")
    with open(importance_path, "w") as f:
        f.write("feature,importance\n")
        for k, v in sorted_importance:
            f.write(f"{k},{v}\n")
    logger.info(f"Saved feature importances to {importance_path}")

    # 13. Save Trained Model wrapper
    model_save_path = os.path.join(experiment_dir, "model.joblib")
    model.save(model_save_path)
    logger.info(f"Saved production model wrapper to {model_save_path}")

    # Register in central registry only when explicitly approved.  A completed
    # experiment is not evidence that a model is safe for trading.
    trainer_reg = Trainer(random_seed=args.seed)
    # Register the model wrapper path
    trainer_reg.registry.register_model(
        model_name=args.model_name,
        version="1.0.0",
        model_path=model_save_path,
        metrics=metrics_json["global"],
        dataset_version=exp_id,
        dataset_hash=diag_report["metrics"].get("dataset_hash", "unknown"),
        feature_registry_version=registry.compute_hash(),
        model_type=args.model_type,
        is_production=args.register_production
    )
    if not args.register_production:
        logger.info("Model was registered as non-production; promote only after out-of-sample backtesting.")

    # 14. Generate and save diagnostic curves/plots
    logger.info("Generating standard performance diagnostic plots...")
    generate_plots(
        y_test_global, y_pred_global, y_proba_global, classes,
        model, feature_cols, X_test_global, experiment_dir
    )

    logger.info("==========================================================")
    logger.info("    PIPELINE EXECUTION SUCCESSFULLY COMPLETED")
    logger.info("==========================================================")
    logger.info(f"Experiment ID: {exp_id}")
    logger.info(f"Model Path   : {model_save_path}")
    logger.info(f"Global Acc   : {global_acc:.4f} | F1-score: {global_f1:.4f}")
    logger.info("==========================================================")


if __name__ == "__main__":
    run_pipeline()
