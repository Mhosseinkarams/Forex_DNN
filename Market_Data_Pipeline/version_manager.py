import os
import re
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger("DatasetVersionManager")

class DatasetVersionManager:
    """
    Manages dataset versions in datasets/v001, datasets/v002, etc.
    Saves parquet, csv, and metadata files atomically to prevent corruption.
    """
    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = output_dir

    def resolve_next_version(self) -> str:
        """Finds the next available version directory under output_dir."""
        if not os.path.exists(self.output_dir):
            return "v001"

        max_v = 0
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                match = re.match(r"^v(\d+)$", item)
                if match:
                    v_num = int(match.group(1))
                    if v_num > max_v:
                        max_v = v_num

        next_v = max_v + 1
        return f"v{next_v:03d}"

    def save_version(
        self,
        version: str,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        feature_registry_json: Dict[str, Any],
        engine_versions_json: Dict[str, Any],
        label_config_json: Dict[str, Any],
        statistics_json: Dict[str, Any],
        manifest_json: Dict[str, Any],
        quality_report_html: Optional[str] = None
    ) -> str:
        """Saves all required files for this dataset version atomically using temporary swaps."""
        version_dir = os.path.join(self.output_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        # File paths
        parquet_path = os.path.join(version_dir, "dataset.parquet")
        csv_path = os.path.join(version_dir, "dataset.csv")
        metadata_path = os.path.join(version_dir, "metadata.json")
        registry_path = os.path.join(version_dir, "feature_registry.json")
        engine_path = os.path.join(version_dir, "engine_versions.json")
        label_path = os.path.join(version_dir, "label_config.json")
        stats_path = os.path.join(version_dir, "statistics.json")
        manifest_path = os.path.join(version_dir, "manifest.json")
        html_path = os.path.join(version_dir, "dataset_quality_report.html")

        # Check if already exists to prevent overwriting
        if os.path.exists(parquet_path):
            logger.warning(f"Version {version} already exists at {version_dir}. Overwriting was requested or occurred.")

        # Helper for atomic dataframe saving
        def save_df_atomically(df_to_save: pd.DataFrame, path: str, as_parquet: bool):
            temp_path = path + ".tmp"
            if as_parquet:
                df_to_save.to_parquet(temp_path, index=False)
            else:
                df_to_save.to_csv(temp_path, index=False)
            os.replace(temp_path, path)

        # Helper for atomic JSON saving
        def save_json_atomically(path: str, data: Dict[str, Any]):
            temp_path = path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)

        # Helper for atomic HTML/text saving
        def save_text_atomically(path: str, content: str):
            temp_path = path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)

        # 1. Save parquet atomically
        save_df_atomically(df, parquet_path, as_parquet=True)
        # 2. Save csv atomically
        save_df_atomically(df, csv_path, as_parquet=False)

        # 3. Save metadata
        save_json_atomically(metadata_path, metadata)
        # 4. Save feature_registry.json
        save_json_atomically(registry_path, feature_registry_json)
        # 5. Save engine_versions.json
        save_json_atomically(engine_path, engine_versions_json)
        # 6. Save label_config.json
        save_json_atomically(label_path, label_config_json)
        # 7. Save statistics.json
        save_json_atomically(stats_path, statistics_json)
        # 8. Save manifest.json
        save_json_atomically(manifest_path, manifest_json)

        # 9. Save dataset_quality_report.html
        if quality_report_html:
            save_text_atomically(html_path, quality_report_html)

        logger.info(f"Successfully saved dataset version {version} atomically in {version_dir}")
        return version_dir
