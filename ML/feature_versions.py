from ML.feature_definition import FeatureDefinition

# Map of versioned features to easily reference or define them
VERSIONED_FEATURES = {
    "ema50_distance": {
        "v1": FeatureDefinition(
            name="ema50_distance_v1",
            display_name="EMA 50 Distance Raw",
            category="Trend",
            dtype=float,
            units="price",
            description="Raw price distance to EMA50",
            source_module="IndicatorEngine",
            version="1.0"
        ),
        "v2": FeatureDefinition(
            name="ema50_distance_v2",
            display_name="EMA 50 Distance ATR Normalized",
            category="Trend",
            dtype=float,
            units="atr",
            description="ATR-normalized price distance to EMA50",
            source_module="IndicatorEngine",
            version="2.0"
        )
    }
}
