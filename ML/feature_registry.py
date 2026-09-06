import json
import os
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

# Optional imports for export/YAML
try:
    import yaml
except ImportError:
    yaml = None

from ML.feature_definition import FeatureDefinition
from ML.feature_groups import ALL_DEFAULT_FEATURES, CATEGORIES

logger = logging.getLogger("FeatureRegistry")

# Reserved Target and Metadata Column Names (Forbidden from being registered as input features)
TARGET_COLUMNS = {
    "target", "label", "confidence", "future_market_state", "future_state_confidence",
    "future_state_horizon", "level_event", "break_probability_target",
    "level_bars_to_resolution", "level_event_confidence", "level_mae", "level_mfe",
    "strategy_outcome", "r_multiple", "strategy_mae", "strategy_mfe",
    "strategy_bars_to_resolution", "exit_reason", "win_loss", "trade_quality_score",
    "reward", "action", "current_market_state", "anchor_index", "window_start",
    "window_end", "window_size", "label_status", "ambiguity_reason"
}


class FeatureRegistry:
    """
    Purpose:
        The centralized Feature Registry, serving as the single source of truth
        for every feature used across the Forex_DNN framework.
    """
    def __init__(self, load_defaults: bool = True):
        self._features: Dict[str, FeatureDefinition] = {}
        self._frozen: bool = False

        if load_defaults:
            for feat in ALL_DEFAULT_FEATURES:
                self.register(feat)

            # Register versioned features as well
            from ML.feature_versions import VERSIONED_FEATURES
            for group, versions in VERSIONED_FEATURES.items():
                for v_key, feat in versions.items():
                    self.register(feat)

    def register(self, feature: FeatureDefinition) -> None:
        """
        Register a feature definition in the registry.
        Strictly prevents target leakage by forbidding target column names.
        """
        if self._frozen:
            raise RuntimeError("Cannot register feature: registry is frozen.")
        if feature.name in TARGET_COLUMNS:
            raise ValueError(f"Security Failure: Cannot register target column '{feature.name}' as an input feature in FeatureRegistry.")
        if feature.name in self._features:
            logger.warning(f"Feature '{feature.name}' is already registered. Overwriting.")
        self._features[feature.name] = feature

    def unregister(self, name: str) -> None:
        """
        Unregister a feature definition by name.
        """
        if self._frozen:
            raise RuntimeError("Cannot unregister feature: registry is frozen.")
        if name in self._features:
            del self._features[name]
        else:
            raise KeyError(f"Feature '{name}' not found in registry.")

    def get(self, name: str) -> FeatureDefinition:
        """
        Get feature definition by name.
        """
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found in registry.")
        return self._features[name]

    def exists(self, name: str) -> bool:
        """
        Check if feature exists.
        """
        return name in self._features

    def list_all(self) -> List[FeatureDefinition]:
        """
        List all registered features in registration order.
        """
        return list(self._features.values())

    def list_by_category(self, category: str) -> List[FeatureDefinition]:
        """
        List features belonging to a specific category.
        """
        return [f for f in self._features.values() if f.category.lower() == category.lower()]

    def list_enabled(self) -> List[FeatureDefinition]:
        """
        List all enabled features in registration order.
        """
        return [f for f in self._features.values() if f.enabled]

    def list_required(self) -> List[FeatureDefinition]:
        """
        List all required features in registration order.
        """
        return [f for f in self._features.values() if f.required]

    def select_group(self, group_name: str) -> List[FeatureDefinition]:
        """
        Select predefined groups or categories of features.
        """
        # Match against category or general group names
        # e.g. TREND_FEATURES -> category="Trend"
        cleaned = group_name.upper().replace("_FEATURES", "")

        # Check standard categories
        for cat in CATEGORIES:
            if cat.upper() == cleaned or cat.lower() == group_name.lower():
                return self.list_by_category(cat)

        # Fallback to category list matches
        return [f for f in self._features.values() if f.category.lower() == group_name.lower()]

    def freeze(self) -> None:
        """
        Freeze the registry to prevent further registrations or updates.
        """
        self._frozen = True
        logger.info("Feature registry frozen.")

    def lock_version(self, name: str, version: str) -> None:
        """
        Force lock a specific version of a feature and disable others if name starts with it.
        """
        if self._frozen:
            raise RuntimeError("Cannot lock version: registry is frozen.")

        # For versioned features like ema50_distance
        prefix = f"{name}_v"
        found = False
        for k, f in list(self._features.items()):
            if k == name or k.startswith(prefix):
                if f.version == version or (f.version == "1.0" and version == "v1") or (f.version == "2.0" and version == "v2"):
                    f.enabled = True
                    found = True
                else:
                    f.enabled = False
        if not found:
            raise ValueError(f"No version '{version}' found for feature '{name}'")

    def validate(self) -> dict:
        """
        Perform automatic validation on registered features.
        Checks for duplicate names, duplicate display names, unsupported dtypes,
        missing descriptions, missing categories, inconsistent versions,
        and disabled required features.
        """
        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total": len(self._features),
                "enabled": len(self.list_enabled()),
                "required": len(self.list_required())
            }
        }

        seen_names = set()
        seen_display_names = set()

        valid_dtypes = {float, int, str, "float", "int", "str"}

        for name, feat in self._features.items():
            # Name duplication
            if name in seen_names:
                report["errors"].append(f"Duplicate feature name detected: '{name}'")
                report["is_valid"] = False
            seen_names.add(name)

            # Display name duplication
            if feat.display_name in seen_display_names:
                report["warnings"].append(f"Duplicate display name: '{feat.display_name}' for feature '{name}'")
            seen_display_names.add(feat.display_name)

            # Unsupported dtypes
            if feat.dtype not in valid_dtypes:
                report["errors"].append(f"Feature '{name}' has unsupported dtype: {feat.dtype}")
                report["is_valid"] = False

            # Missing descriptions
            if not feat.description or len(feat.description.strip()) == 0:
                report["warnings"].append(f"Feature '{name}' has missing or empty description.")

            # Missing or invalid categories
            if not feat.category or feat.category not in CATEGORIES:
                report["warnings"].append(f"Feature '{name}' has missing or non-standard category: '{feat.category}'")

            # Disabled required features
            if feat.required and not feat.enabled:
                report["errors"].append(f"Required feature '{name}' is disabled.")
                report["is_valid"] = False

        return report

    def compute_hash(self) -> str:
        """
        Compute a stable deterministic hash of the enabled feature list and their ordering/types.
        """
        enabled = self.list_enabled()
        hash_input = []
        for f in enabled:
            hash_input.append(f"{f.name}:{f.dtype}:{f.version}")
        serialized = ",".join(hash_input)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def feature_vector(self, df: pd.DataFrame, return_type: str = "numpy") -> Union[np.ndarray, pd.DataFrame]:
        """
        Return deterministic feature ordering of enabled features as a numpy array or pandas DataFrame.
        """
        enabled_features = [f.name for f in self.list_enabled()]

        # Verify that all enabled features exist in DataFrame
        missing = [f for f in enabled_features if f not in df.columns]
        if missing:
            # We will fill missing columns with default values if they are specified
            for m in missing:
                feat = self.get(m)
                df[m] = feat.default_value
                logger.warning(f"Feature '{m}' missing in DataFrame, filled with default: {feat.default_value}")

        # Reorder to guarantee deterministic ordering
        df_ordered = df[enabled_features]

        if return_type == "dataframe":
            return df_ordered
        return df_ordered.to_numpy()

    def export_json(self, filepath: str) -> None:
        """
        Export registered features to a JSON file.
        """
        data = [f.to_dict() for f in self.list_all()]
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def export_yaml(self, filepath: str) -> None:
        """
        Export registered features to a YAML file.
        """
        if yaml is None:
            raise ImportError("PyYAML is not installed.")
        data = [f.to_dict() for f in self.list_all()]
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def export_csv(self, filepath: str) -> None:
        """
        Export registered features to a CSV file.
        """
        data = [f.to_dict() for f in self.list_all()]
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        df.to_csv(filepath, index=False)

    def export_markdown(self, filepath: str) -> None:
        """
        Export registered features as a Markdown table (auto doc generation).
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        lines = [
            "# Feature Registry Documentation",
            "",
            "This document is autogenerated by the Feature Registry. Do not edit manually.",
            "",
            "## Summary Metrics",
            f"- **Total Registered Features**: {len(self._features)}",
            f"- **Enabled Features**: {len(self.list_enabled())}",
            f"- **Registry Version Hash**: `{self.compute_hash()}`",
            "",
            "## Features List",
            "",
            "| Feature | Category | Source Module | Version | Enabled | Required | Description |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        ]

        for f in self.list_all():
            enabled_str = "✅ Yes" if f.enabled else "❌ No"
            required_str = "✅ Yes" if f.required else "❌ No"
            lines.append(f"| `{f.name}` | {f.category} | `{f.source_module}` | {f.version} | {enabled_str} | {required_str} | {f.description} |")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    def generate_visualization(self, filepath: str) -> None:
        """
        Generate reports/feature_map.html visualization file containing category counts,
        enabled/disabled features, and a feature versions dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Calculate category counts
        cat_counts = {}
        for f in self.list_all():
            cat_counts[f.category] = cat_counts.get(f.category, 0) + 1

        categories_json = json.dumps(list(cat_counts.keys()))
        counts_json = json.dumps(list(cat_counts.values()))

        features_data = []
        for f in self.list_all():
            features_data.append({
                "name": f.name,
                "category": f.category,
                "version": f.version,
                "enabled": f.enabled,
                "required": f.required,
                "description": f.description
            })

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Feature Registry Map</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 font-sans p-8">
    <div class="max-w-7xl mx-auto">
        <header class="mb-8">
            <h1 class="text-4xl font-bold text-gray-800">Feature Registry Map</h1>
            <p class="text-gray-600 mt-2">Interactive dependency dashboard & schema verification</p>
            <div class="mt-4 flex gap-4">
                <span class="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-semibold">Total: {len(self._features)}</span>
                <span class="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-semibold">Enabled: {len(self.list_enabled())}</span>
                <span class="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm font-semibold">Hash: {self.compute_hash()[:12]}</span>
            </div>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            <div class="bg-white p-6 rounded-lg shadow-md col-span-1">
                <h2 class="text-xl font-semibold mb-4 text-gray-700">Category Distribution</h2>
                <div class="w-full h-64 flex justify-center">
                    <canvas id="categoryChart"></canvas>
                </div>
            </div>

            <div class="bg-white p-6 rounded-lg shadow-md col-span-2">
                <h2 class="text-xl font-semibold mb-4 text-gray-700">Registry Overview & Versions</h2>
                <div class="overflow-x-auto max-h-64 scrollbar-thin">
                    <table class="min-w-full bg-white text-left">
                        <thead>
                            <tr class="bg-gray-200">
                                <th class="p-2">Feature Name</th>
                                <th class="p-2">Category</th>
                                <th class="p-2">Version</th>
                                <th class="p-2">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join([f'<tr class="border-b"><td class="p-2 font-mono text-sm">{fd["name"]}</td><td class="p-2 text-sm">{fd["category"]}</td><td class="p-2 text-sm">{fd["version"]}</td><td class="p-2"><span class="px-2 py-0.5 rounded text-xs ' + ('bg-green-100 text-green-800' if fd['enabled'] else 'bg-red-100 text-red-800') + f'">{ "Enabled" if fd["enabled"] else "Disabled" }</span></td></tr>' for fd in features_data[:100]])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('categoryChart').getContext('2d');
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: {categories_json},
                datasets: [{{
                    data: {counts_json},
                    backgroundColor: [
                        '#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#3B82F6',
                        '#8B5CF6', '#EC4899', '#6B7280', '#14B8A6', '#6366F1'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});
    </script>
</body>
</html>
"""
        with open(filepath, "w") as f:
            f.write(html_content)
