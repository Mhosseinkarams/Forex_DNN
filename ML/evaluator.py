import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from ML.base_model import BaseTradingModel

logger = logging.getLogger("Evaluator")


class Evaluator:
    """
    Centralized Evaluation Engine for BaseTradingModel subclasses.
    Calculates statistical benchmarks, calibration diagnostics, and outputs
    both detailed markdown reports and rich interactive HTML files.
    """
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_and_report(
        self,
        model: BaseTradingModel,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        classes: List[str],
        report_name: str = "model_evaluation_report"
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation, prints results, saves static matplotlib figures
        (if matplotlib is available), and writes fully descriptive Markdown and interactive HTML reports.
        """
        inference_engine = model.calibrated_model if model.calibrated_model is not None else model.model
        y_pred = inference_engine.predict(X_val)
        y_proba = inference_engine.predict_proba(X_val)

        # Baseline stats
        acc = accuracy_score(y_val, y_pred)
        is_binary = len(classes) <= 2

        if is_binary:
            prec = precision_score(y_val, y_pred, average="binary", zero_division=0)
            rec = recall_score(y_val, y_pred, average="binary", zero_division=0)
            f1 = f1_score(y_val, y_pred, average="binary", zero_division=0)
            auc = roc_auc_score(y_val, y_proba[:, 1]) if y_proba.ndim == 2 else 0.5
        else:
            prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
            auc = 0.0  # Multiclass AUC not computed by default here

        cm = confusion_matrix(y_val, y_pred, labels=list(range(len(classes))))

        # Class distribution
        unique, counts = np.unique(y_val, return_counts=True)
        dist = {classes[int(u)]: int(c) for u, c in zip(unique, counts) if int(u) < len(classes)}

        # Calibration curve (on first active class)
        prob_true, prob_pred = [], []
        if is_binary and y_proba.ndim == 2:
            prob_true, prob_pred = calibration_curve(y_val, y_proba[:, 1], n_bins=10)
            prob_true = prob_true.tolist()
            prob_pred = prob_pred.tolist()

        # Feature importances
        importance_map = model.get_feature_importance()
        sorted_importance = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)[:15]

        report_data = {
            "model_name": model.__class__.__name__,
            "model_type": model.model_type,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "class_distribution": dist,
            "feature_importance": {k: float(v) for k, v in sorted_importance},
            "calibration": {
                "prob_true": prob_true,
                "prob_pred": prob_pred
            }
        }

        # Generate markdown report
        self._write_markdown_report(report_name, report_data, classes)

        # Generate HTML report
        self._write_html_report(report_name, report_data, classes)

        # Generate matplotlib plots if available
        if plt is not None:
            self._save_static_plots(report_name, report_data, classes)

        logger.info(f"Generated comprehensive reports under {self.output_dir}/{report_name}.*")
        return report_data

    def _write_markdown_report(self, report_name: str, data: Dict[str, Any], classes: List[str]):
        filepath = os.path.join(self.output_dir, f"{report_name}.md")
        lines = [
            f"# Performance & Verification Audit: {data['model_name']}",
            "",
            f"- **Backend Learning Engine**: {data['model_type'].upper()}",
            f"- **Timestamp**: {pd.Timestamp.now().isoformat()}",
            "",
            "## Summary Metrics",
            f"- **Accuracy**: {data['accuracy']:.4f}",
            f"- **Precision**: {data['precision']:.4f}",
            f"- **Recall**: {data['recall']:.4f}",
            f"- **F1-Score**: {data['f1_score']:.4f}",
        ]
        if len(classes) <= 2:
            lines.append(f"- **ROC-AUC**: {data['roc_auc']:.4f}")

        lines.extend([
            "",
            "## Class Distributions",
        ])
        for k, v in data["class_distribution"].items():
            lines.append(f"- **{k}**: {v} samples")

        lines.extend([
            "",
            "## Top Feature Importances",
            "| Feature Name | Relative Importance Score |",
            "| :--- | :--- |"
        ])
        for k, v in data["feature_importance"].items():
            lines.append(f"| `{k}` | {v:.6f} |")

        lines.extend([
            "",
            "## Confusion Matrix",
            f"Target classes order: {classes}",
            "```"
        ])
        # Format Confusion Matrix prettily
        cm = np.array(data["confusion_matrix"])
        lines.append(str(cm))
        lines.append("```")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    def _write_html_report(self, report_name: str, data: Dict[str, Any], classes: List[str]):
        filepath = os.path.join(self.output_dir, f"{report_name}.html")

        # Serialize stats to pass to Chart.js
        features = list(data["feature_importance"].keys())
        importances = list(data["feature_importance"].values())
        dist_labels = list(data["class_distribution"].keys())
        dist_values = list(data["class_distribution"].values())

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Report: {data['model_name']}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 font-sans p-8">
    <div class="max-w-6xl mx-auto">
        <header class="mb-10 bg-white p-6 rounded-lg shadow-sm border border-gray-100">
            <h1 class="text-3xl font-extrabold text-gray-800">Model Evaluation & Audit</h1>
            <p class="text-gray-500 mt-1">Model Name: <span class="font-mono text-indigo-600 font-semibold">{data['model_name']}</span> | Backend: <span class="font-mono text-indigo-600 font-semibold">{data['model_type'].upper()}</span></p>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                <div class="bg-indigo-50 p-4 rounded text-center">
                    <span class="block text-xs font-semibold text-indigo-600 uppercase">Accuracy</span>
                    <span class="text-2xl font-bold text-indigo-900">{data['accuracy']:.4f}</span>
                </div>
                <div class="bg-green-50 p-4 rounded text-center">
                    <span class="block text-xs font-semibold text-green-600 uppercase">Precision</span>
                    <span class="text-2xl font-bold text-green-900">{data['precision']:.4f}</span>
                </div>
                <div class="bg-blue-50 p-4 rounded text-center">
                    <span class="block text-xs font-semibold text-blue-600 uppercase">Recall</span>
                    <span class="text-2xl font-bold text-blue-900">{data['recall']:.4f}</span>
                </div>
                <div class="bg-purple-50 p-4 rounded text-center">
                    <span class="block text-xs font-semibold text-purple-600 uppercase">F1-Score</span>
                    <span class="text-2xl font-bold text-purple-900">{data['f1_score']:.4f}</span>
                </div>
            </div>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <!-- Feature Importance Chart -->
            <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
                <h2 class="text-lg font-bold text-gray-700 mb-4">Top Feature Importances</h2>
                <div class="h-64">
                    <canvas id="importanceChart"></canvas>
                </div>
            </div>

            <!-- Class Distribution Chart -->
            <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
                <h2 class="text-lg font-bold text-gray-700 mb-4">Validation Class Distribution</h2>
                <div class="h-64 flex justify-center">
                    <canvas id="distributionChart"></canvas>
                </div>
            </div>
        </div>

        <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-100 mb-8">
            <h2 class="text-lg font-bold text-gray-700 mb-4">Confusion Matrix</h2>
            <div class="overflow-x-auto">
                <table class="min-w-full text-center border-collapse">
                    <thead>
                        <tr class="bg-gray-100 border-b border-gray-200">
                            <th class="p-3 text-sm font-semibold text-gray-600">Actual \\ Predicted</th>
                            {"".join([f'<th class="p-3 text-sm font-semibold text-gray-600">{c}</th>' for c in classes])}
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([
                            f'<tr class="border-b border-gray-100"><td class="p-3 font-semibold text-gray-600 text-left">{classes[i]}</td>' +
                            "".join([f'<td class="p-3 text-gray-800 ' + ('bg-indigo-50 font-bold' if i==j else '') + f'">{val}</td>' for j, val in enumerate(row)]) +
                            '</tr>'
                            for i, row in enumerate(data['confusion_matrix'])
                        ])}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Importance Chart
        new Chart(document.getElementById('importanceChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(features)},
                datasets: [{{
                    label: 'Importance Score',
                    data: {json.dumps(importances)},
                    backgroundColor: '#4F46E5',
                    borderRadius: 4
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }}
            }}
        }});

        // Distribution Chart
        new Chart(document.getElementById('distributionChart'), {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(dist_labels)},
                datasets: [{{
                    data: {json.dumps(dist_values)},
                    backgroundColor: ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
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
            f.write(html)

    def _save_static_plots(self, report_name: str, data: Dict[str, Any], classes: List[str]):
        """
        Saves a static compilation png plot of features & distributions.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Feature Importance Plot
        feats = list(data["feature_importance"].keys())[::-1]
        imps = list(data["feature_importance"].values())[::-1]
        axes[0].barh(feats, imps, color="indigo")
        axes[0].set_title("Relative Feature Importance")
        axes[0].set_xlabel("Relative Importance Score")

        # 2. Confusion Matrix Visualizer
        cm = np.array(data["confusion_matrix"])
        im = axes[1].imshow(cm, cmap="Purples")
        axes[1].set_title("Confusion Matrix Heatmap")
        axes[1].set_xticks(np.arange(len(classes)))
        axes[1].set_yticks(np.arange(len(classes)))
        axes[1].set_xticklabels(classes)
        axes[1].set_yticklabels(classes)

        # Loop over data dimensions and create text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{report_name}.png")
        plt.savefig(plot_path)
        plt.close()
