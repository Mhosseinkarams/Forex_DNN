import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from ML.feature_definition import FeatureDefinition
from ML.feature_registry import FeatureRegistry

class TestFeatureRegistry(unittest.TestCase):
    def setUp(self):
        # Create registry with load_defaults=False for isolated testing
        self.registry = FeatureRegistry(load_defaults=False)

        self.feat_a = FeatureDefinition(
            name="feature_a",
            display_name="Feature A",
            category="Trend",
            dtype=float,
            units="pips",
            description="Test Feature A",
            source_module="IndicatorEngine"
        )

        self.feat_b = FeatureDefinition(
            name="feature_b",
            display_name="Feature B",
            category="Volatility",
            dtype=int,
            units="count",
            description="Test Feature B",
            source_module="IndicatorEngine",
            required=False,
            enabled=True
        )

    def test_register_and_get(self):
        self.registry.register(self.feat_a)
        self.assertTrue(self.registry.exists("feature_a"))

        fetched = self.registry.get("feature_a")
        self.assertEqual(fetched.display_name, "Feature A")
        self.assertEqual(fetched.category, "Trend")

    def test_unregister(self):
        self.registry.register(self.feat_a)
        self.registry.unregister("feature_a")
        self.assertFalse(self.registry.exists("feature_a"))

        with self.assertRaises(KeyError):
            self.registry.unregister("non_existent")

    def test_list_helpers(self):
        self.registry.register(self.feat_a)
        self.registry.register(self.feat_b)

        all_features = self.registry.list_all()
        self.assertEqual(len(all_features), 2)

        enabled = self.registry.list_enabled()
        self.assertEqual(len(enabled), 2)

        required = self.registry.list_required()
        self.assertEqual(len(required), 1)
        self.assertEqual(required[0].name, "feature_a")

    def test_select_group(self):
        self.registry.register(self.feat_a)
        self.registry.register(self.feat_b)

        trend_group = self.registry.select_group("Trend")
        self.assertEqual(len(trend_group), 1)
        self.assertEqual(trend_group[0].name, "feature_a")

    def test_freeze(self):
        self.registry.register(self.feat_a)
        self.registry.freeze()

        with self.assertRaises(RuntimeError):
            self.registry.register(self.feat_b)

        with self.assertRaises(RuntimeError):
            self.registry.unregister("feature_a")

    def test_lock_version(self):
        # Register two versions of same feature prefix
        v1 = FeatureDefinition(
            name="ema_distance_v1",
            display_name="EMA Distance V1",
            category="Trend",
            dtype=float,
            units="price",
            description="V1 description",
            source_module="IndicatorEngine",
            version="1.0"
        )
        v2 = FeatureDefinition(
            name="ema_distance_v2",
            display_name="EMA Distance V2",
            category="Trend",
            dtype=float,
            units="atr",
            description="V2 description",
            source_module="IndicatorEngine",
            version="2.0"
        )
        self.registry.register(v1)
        self.registry.register(v2)

        # Lock to v2
        self.registry.lock_version("ema_distance", "2.0")

        self.assertTrue(self.registry.get("ema_distance_v2").enabled)
        self.assertFalse(self.registry.get("ema_distance_v1").enabled)

    def test_deterministic_feature_vector(self):
        self.registry.register(self.feat_b)
        self.registry.register(self.feat_a) # registered B first, then A

        # Ordering must match registration order: [feat_b, feat_a]
        df = pd.DataFrame({
            "feature_a": [1.5, 2.5],
            "feature_b": [10, 20]
        })

        vec = self.registry.feature_vector(df, return_type="numpy")
        # Row 1: [feature_b_val, feature_a_val] -> [10, 1.5]
        np.testing.assert_array_equal(vec[0], [10.0, 1.5])
        np.testing.assert_array_equal(vec[1], [20.0, 2.5])

        # Test dataframe return type
        df_vec = self.registry.feature_vector(df, return_type="dataframe")
        self.assertEqual(list(df_vec.columns), ["feature_b", "feature_a"])

    def test_validate(self):
        self.registry.register(self.feat_a)
        report = self.registry.validate()
        self.assertTrue(report["is_valid"])
        self.assertEqual(len(report["errors"]), 0)

        # Introduce violation: disabled required feature
        self.feat_a.enabled = False
        report = self.registry.validate()
        self.assertFalse(report["is_valid"])
        self.assertEqual(len(report["errors"]), 1)

        # Restore state
        self.feat_a.enabled = True

    def test_compute_hash(self):
        self.registry.register(self.feat_a)
        self.registry.register(self.feat_b)

        hash_1 = self.registry.compute_hash()

        # Create identical registry and assert hash match
        other_reg = FeatureRegistry(load_defaults=False)
        other_reg.register(self.feat_a)
        other_reg.register(self.feat_b)

        self.assertEqual(hash_1, other_reg.compute_hash())

if __name__ == "__main__":
    unittest.main()
