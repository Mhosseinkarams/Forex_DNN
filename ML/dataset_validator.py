import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger("DatasetValidator")


class DatasetValidator:
    """
    Performs comprehensive validation checks on generated, rule-labeled machine learning datasets.
    Helps ensure that feature vectors, target labels, and metadata are correct and suitable
    for classifier training.
    """
    def __init__(self, expected_window_size: Optional[int] = None):
        self.expected_window_size = expected_window_size

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs all validation checks on the dataset.

        Returns:
            Dict containing validation status, details of checks, lists of errors and warnings.
        """
        logger.info(f"Starting dataset validation on {len(df)} samples...")

        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "checks": {
                "missing_values": "PASSED",
                "duplicate_samples": "PASSED",
                "market_structure_outputs": "PASSED",
                "supply_demand_outputs": "PASSED",
                "timestamp_consistency": "PASSED",
                "window_consistency": "PASSED",
                "class_distribution": "PASSED"
            }
        }

        if df.empty:
            report["is_valid"] = False
            report["errors"].append("The dataset is empty.")
            for check in report["checks"]:
                report["checks"][check] = "FAILED"
            return report

        # 1. Check for Missing Values
        # Allow missing value in reason or description columns, but feature and target columns should be clean.
        target_cols = ["label", "confidence"]
        metadata_cols = ["symbol", "timeframe", "window_start", "window_end", "datetime"]

        # Identify feature columns (everything not target, metadata, or helper/debugging columns)
        non_feature_cols = target_cols + metadata_cols + ["label_reason", "label_version", "engine_version"]
        feature_cols = [c for c in df.columns if c not in non_feature_cols]

        # Check target/metadata
        for col in target_cols + metadata_cols:
            if col in df.columns:
                n_null = df[col].isnull().sum()
                if n_null > 0:
                    report["errors"].append(f"Column '{col}' has {n_null} missing/null values.")
                    report["checks"]["missing_values"] = "FAILED"
                    report["is_valid"] = False

        # Check feature columns
        missing_features = []
        for col in feature_cols:
            n_null = df[col].isnull().sum()
            if n_null > 0:
                missing_features.append((col, n_null))

        if missing_features:
            report["errors"].append(f"Feature columns contain missing values: {missing_features}")
            report["checks"]["missing_values"] = "FAILED"
            report["is_valid"] = False

        # 2. Check for Duplicate Samples
        # Rows shouldn't share both exact feature vectors AND exact timestamps
        if "datetime" in df.columns:
            n_dup_time = df.duplicated(subset=["datetime"]).sum()
            if n_dup_time > 0:
                report["warnings"].append(f"Found {n_dup_time} samples sharing duplicate timestamps/datetimes.")
                report["checks"]["duplicate_samples"] = "WARNING"

            n_dup_rows = df.duplicated().sum()
            if n_dup_rows > 0:
                report["errors"].append(f"Found {n_dup_rows} exact duplicate rows in the dataset.")
                report["checks"]["duplicate_samples"] = "FAILED"
                report["is_valid"] = False

        # 3. Check for Invalid Market Structure Outputs
        structure_cols_non_negative = ["bos_count_last_n", "choch_count_last_n", "time_since_last_bos", "time_since_last_choch"]
        for col in structure_cols_non_negative:
            if col in df.columns:
                # Exclude the default -1 or 999.0 fallbacks if standard, but actual negatives (other than fallback) are bad
                # Fallback values like 999 or -1 might be used, check if there are invalid/unreasonable negatives below -1
                invalid_vals = (df[col] < -1.0).sum()
                if invalid_vals > 0:
                    report["errors"].append(f"Column '{col}' contains {invalid_vals} invalid negative values.")
                    report["checks"]["market_structure_outputs"] = "FAILED"
                    report["is_valid"] = False

        # 4. Check for Invalid Supply/Demand Outputs
        sd_cols = ["supply_distance", "demand_distance"]
        for col in sd_cols:
            if col in df.columns:
                # Allowed is positive value or -1 (no active zone fallback)
                invalid_vals = ((df[col] < 0.0) & (df[col] != -1.0)).sum()
                if invalid_vals > 0:
                    report["errors"].append(f"Column '{col}' has {invalid_vals} invalid values (negative but not -1).")
                    report["checks"]["supply_demand_outputs"] = "FAILED"
                    report["is_valid"] = False

        # 5. Check for Timestamp Consistency
        if "datetime" in df.columns:
            try:
                times = pd.to_datetime(df["datetime"])
                is_monotonic = times.is_monotonic_increasing
                if not is_monotonic:
                    report["errors"].append("Timestamps/datetimes are not strictly chronologically ordered (monotonic increasing).")
                    report["checks"]["timestamp_consistency"] = "FAILED"
                    report["is_valid"] = False
            except Exception as e:
                report["errors"].append(f"Failed to validate timestamp consistency: {e}")
                report["checks"]["timestamp_consistency"] = "FAILED"
                report["is_valid"] = False

        # 6. Check for Window Consistency
        if "window_start" in df.columns and "window_end" in df.columns:
            diffs = df["window_end"] - df["window_start"] + 1
            if (diffs <= 0).any():
                report["errors"].append("Found invalid windows where window_end <= window_start.")
                report["checks"]["window_consistency"] = "FAILED"
                report["is_valid"] = False

            # If expected_window_size is specified, verify all windows match this size
            if self.expected_window_size is not None:
                mismatches = (diffs != self.expected_window_size).sum()
                if mismatches > 0:
                    report["errors"].append(f"Found {mismatches} windows whose size does not match the expected window_size of {self.expected_window_size}.")
                    report["checks"]["window_consistency"] = "FAILED"
                    report["is_valid"] = False

            # Check stride/stride gaps are positive
            if len(df) > 1:
                start_diffs = df["window_start"].diff().dropna()
                if (start_diffs < 0).any():
                    report["errors"].append("Window starts are not monotonically increasing.")
                    report["checks"]["window_consistency"] = "FAILED"
                    report["is_valid"] = False

        # 7. Check Class Distribution
        if "label" in df.columns:
            counts = df["label"].value_counts()
            # If any class contains 0 items, warn or fail
            classes_needed = ["TREND", "RANGE", "TRANSITION"]
            for cls in classes_needed:
                if cls not in counts or counts[cls] == 0:
                    report["warnings"].append(f"Class '{cls}' has zero samples in the generated dataset.")
                    report["checks"]["class_distribution"] = "WARNING"

            # Check if there is massive class imbalance (e.g. one class is > 95% of dataset)
            total = len(df)
            for cls, cnt in counts.items():
                ratio = cnt / total
                if ratio > 0.95:
                    report["warnings"].append(f"Class '{cls}' comprises {ratio:.1%} of the dataset (extreme class imbalance).")
                    report["checks"]["class_distribution"] = "WARNING"

        logger.info(f"Dataset validation completed. Status: {'PASSED' if report['is_valid'] else 'FAILED'}")
        return report
