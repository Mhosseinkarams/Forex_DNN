from ML.feature_definition import FeatureDefinition

# Categories definitions
CATEGORIES = [
    "Trend", "Momentum", "Structure", "SupplyDemand", "PriceAction",
    "Volatility", "Risk", "Time", "Session", "ML", "Trade", "Account",
    "Position", "Statistics", "Custom"
]

# Predefined groups
TREND_FEATURES = [
    FeatureDefinition(
        name="ema50_slope",
        display_name="EMA 50 Slope",
        category="Trend",
        dtype=float,
        units="price/bar",
        description="Slope of EMA50 over previous bar",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="ema600_slope",
        display_name="EMA 600 Slope",
        category="Trend",
        dtype=float,
        units="price/bar",
        description="Slope of EMA600 over previous bar",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="ema800_slope",
        display_name="EMA 800 Slope",
        category="Trend",
        dtype=float,
        units="price/bar",
        description="Slope of EMA800 over previous bar",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="ema_separation",
        display_name="EMA Separation",
        category="Trend",
        dtype=float,
        units="pips",
        description="Distance between EMA50 and EMA600/800",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="ema_compression",
        display_name="EMA Compression",
        category="Trend",
        dtype=float,
        units="ratio",
        description="Compression metric of EMAs indicating convergence",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="distance_to_ema50",
        display_name="Distance to EMA50",
        category="Trend",
        dtype=float,
        units="atr",
        description="Distance from close to EMA50 in ATR units",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="distance_to_ema600",
        display_name="Distance to EMA600",
        category="Trend",
        dtype=float,
        units="atr",
        description="Distance from close to EMA600 in ATR units",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="distance_to_ema800",
        display_name="Distance to EMA800",
        category="Trend",
        dtype=float,
        units="atr",
        description="Distance from close to EMA800 in ATR units",
        source_module="IndicatorEngine"
    )
]

MOMENTUM_FEATURES = [
    FeatureDefinition(
        name="candle_body",
        display_name="Candle Body",
        category="Momentum",
        dtype=float,
        units="ratio",
        description="Candle body size relative to total high-low range",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="upper_wick",
        display_name="Upper Wick Size",
        category="Momentum",
        dtype=float,
        units="ratio",
        description="Upper wick size relative to total high-low range",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="lower_wick",
        display_name="Lower Wick Size",
        category="Momentum",
        dtype=float,
        units="ratio",
        description="Lower wick size relative to total high-low range",
        source_module="IndicatorEngine"
    )
]

STRUCTURE_FEATURES = [
    FeatureDefinition(
        name="bos_count_last_n",
        display_name="BOS Count Last N",
        category="Structure",
        dtype=int,
        units="count",
        description="Number of BOS detections in the last N bars",
        source_module="MarketStructureEngine"
    ),
    FeatureDefinition(
        name="choch_count_last_n",
        display_name="CHOCH Count Last N",
        category="Structure",
        dtype=int,
        units="count",
        description="Number of CHOCH detections in the last N bars",
        source_module="MarketStructureEngine"
    ),
    FeatureDefinition(
        name="time_since_last_bos",
        display_name="Time Since Last BOS",
        category="Structure",
        dtype=float,
        units="bars",
        description="Bars since the most recent BOS occurrence",
        source_module="MarketStructureEngine"
    ),
    FeatureDefinition(
        name="time_since_last_choch",
        display_name="Time Since Last CHOCH",
        category="Structure",
        dtype=float,
        units="bars",
        description="Bars since the most recent CHOCH occurrence",
        source_module="MarketStructureEngine"
    ),
    FeatureDefinition(
        name="bos_direction",
        display_name="BOS Direction",
        category="Structure",
        dtype=int,
        units="direction",
        description="Direction of the last BOS (1 for bullish, -1 for bearish, 0 for none)",
        source_module="MarketStructureEngine"
    ),
    FeatureDefinition(
        name="choch_direction",
        display_name="CHOCH Direction",
        category="Structure",
        dtype=int,
        units="direction",
        description="Direction of the last CHOCH (1 for bullish, -1 for bearish, 0 for none)",
        source_module="MarketStructureEngine"
    ),
    FeatureDefinition(
        name="protected_high_distance",
        display_name="Protected High Distance",
        category="Structure",
        dtype=float,
        units="atr",
        description="Distance to the protected high level in ATR units",
        source_module="MarketStructureEngine"
    ),
    FeatureDefinition(
        name="protected_low_distance",
        display_name="Protected Low Distance",
        category="Structure",
        dtype=float,
        units="atr",
        description="Distance to the protected low level in ATR units",
        source_module="MarketStructureEngine"
    )
]

SUPPLY_DEMAND_FEATURES = [
    FeatureDefinition(
        name="supply_distance",
        display_name="Supply Distance",
        category="SupplyDemand",
        dtype=float,
        units="atr",
        description="ATR normalized distance to the nearest supply zone",
        source_module="SupplyDemandEngine"
    ),
    FeatureDefinition(
        name="supply_width",
        display_name="Supply Zone Width",
        category="SupplyDemand",
        dtype=float,
        units="pips",
        description="Width of the nearest supply zone",
        source_module="SupplyDemandEngine"
    ),
    FeatureDefinition(
        name="supply_strength",
        display_name="Supply Zone Strength",
        category="SupplyDemand",
        dtype=float,
        units="score",
        description="Strength score of the nearest supply zone",
        source_module="SupplyDemandEngine"
    ),
    FeatureDefinition(
        name="supply_freshness",
        display_name="Supply Zone Freshness",
        category="SupplyDemand",
        dtype=int,
        units="binary",
        description="Freshness flag (1 for fresh, 0 for mitigated/broken)",
        source_module="SupplyDemandEngine"
    ),
    FeatureDefinition(
        name="supply_touch_count",
        display_name="Supply Zone Touch Count",
        category="SupplyDemand",
        dtype=int,
        units="count",
        description="Touch count of the nearest supply zone",
        source_module="SupplyDemandEngine"
    ),
    FeatureDefinition(
        name="demand_distance",
        display_name="Demand Distance",
        category="SupplyDemand",
        dtype=float,
        units="atr",
        description="ATR normalized distance to the nearest demand zone",
        source_module="SupplyDemandEngine"
    ),
    FeatureDefinition(
        name="demand_width",
        display_name="Demand Zone Width",
        category="SupplyDemand",
        dtype=float,
        units="pips",
        description="Width of the nearest demand zone",
        source_module="SupplyDemandEngine"
    ),
    FeatureDefinition(
        name="demand_strength",
        display_name="Demand Zone Strength",
        category="SupplyDemand",
        dtype=float,
        units="score",
        description="Strength score of the nearest demand zone",
        source_module="SupplyDemandEngine"
    ),
    FeatureDefinition(
        name="demand_freshness",
        display_name="Demand Zone Freshness",
        category="SupplyDemand",
        dtype=int,
        units="binary",
        description="Freshness flag (1 for fresh, 0 for mitigated/broken)",
        source_module="SupplyDemandEngine"
    ),
    FeatureDefinition(
        name="demand_touch_count",
        display_name="Demand Zone Touch Count",
        category="SupplyDemand",
        dtype=int,
        units="count",
        description="Touch count of the nearest demand zone",
        source_module="SupplyDemandEngine"
    )
]

PRICE_ACTION_FEATURES = [
    FeatureDefinition(
        name="candle_range",
        display_name="Candle Range",
        category="PriceAction",
        dtype=float,
        units="atr",
        description="Total candle height (High - Low) in ATR units",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="volume",
        display_name="Tick Volume",
        category="PriceAction",
        dtype=float,
        units="count",
        description="Volume of ticks in the bar",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="spread",
        display_name="Bid-Ask Spread",
        category="PriceAction",
        dtype=float,
        units="pips",
        description="Current spread in pips",
        source_module="IndicatorEngine"
    )
]

VOLATILITY_FEATURES = [
    FeatureDefinition(
        name="atr",
        display_name="ATR",
        category="Volatility",
        dtype=float,
        units="pips",
        description="Average True Range (ATR)",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="atr_percentile",
        display_name="ATR Percentile",
        category="Volatility",
        dtype=float,
        units="ratio",
        description="ATR rank percentile over a rolling window",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="atr_ratio",
        display_name="ATR Ratio",
        category="Volatility",
        dtype=float,
        units="ratio",
        description="Ratio of fast ATR to slow ATR",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="realized_volatility",
        display_name="Realized Volatility",
        category="Volatility",
        dtype=float,
        units="ratio",
        description="Realized volatility computed from intraday returns",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="rolling_std",
        display_name="Rolling Standard Deviation",
        category="Volatility",
        dtype=float,
        units="price",
        description="Rolling standard deviation of close prices",
        source_module="IndicatorEngine"
    )
]

SESSION_FEATURES = [
    FeatureDefinition(
        name="hour",
        display_name="Hour of Day",
        category="Time",
        dtype=int,
        units="hour",
        description="Hour of day (0-23)",
        source_module="IndicatorEngine",
        normalize=False
    ),
    FeatureDefinition(
        name="weekday",
        display_name="Day of Week",
        category="Time",
        dtype=int,
        units="day",
        description="Day of week (0-4 for Mon-Fri)",
        source_module="IndicatorEngine",
        normalize=False
    ),
    FeatureDefinition(
        name="session",
        display_name="Trading Session",
        category="Session",
        dtype=str,
        units="none",
        description="Identified trading session (Asian, London, NewYork, or London/NY)",
        source_module="IndicatorEngine",
        normalize=False
    )
]

REGIME_FEATURES = [
    FeatureDefinition(
        name="trend_score",
        display_name="Trend Score",
        category="Trend",
        dtype=float,
        units="ratio",
        description="Linear regression r-squared trend confirmation score",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="range_score",
        display_name="Range Score",
        category="PriceAction",
        dtype=float,
        units="ratio",
        description="Ratio of high-low range over the period to total path traveled",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="compression_score",
        display_name="Compression Score",
        category="Volatility",
        dtype=float,
        units="ratio",
        description="Degree of candle range contraction relative to previous period",
        source_module="IndicatorEngine"
    )
]

POSITION_FEATURES = [
    FeatureDefinition(
        name="distance_to_nearest_high",
        display_name="Distance to Nearest High",
        category="Position",
        dtype=float,
        units="atr",
        description="Distance to nearest local high level in ATR units",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="distance_to_nearest_low",
        display_name="Distance to Nearest Low",
        category="Position",
        dtype=float,
        units="atr",
        description="Distance to nearest local low level in ATR units",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="distance_to_structure_break",
        display_name="Distance to Structure Break",
        category="Position",
        dtype=float,
        units="atr",
        description="Distance to nearest BOS or CHOCH level in ATR units",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="distance_to_invalidation_level",
        display_name="Distance to Invalidation Level",
        category="Position",
        dtype=float,
        units="atr",
        description="Distance to the invalidation level (stop loss) in ATR units",
        source_module="IndicatorEngine"
    ),
    FeatureDefinition(
        name="risk_reward_estimate",
        display_name="Risk Reward Estimate",
        category="Position",
        dtype=float,
        units="ratio",
        description="Estimated risk reward ratio based on structural levels",
        source_module="TradeLocationEngine"
    )
]

ALL_DEFAULT_FEATURES = (
    TREND_FEATURES + MOMENTUM_FEATURES + STRUCTURE_FEATURES +
    SUPPLY_DEMAND_FEATURES + PRICE_ACTION_FEATURES + VOLATILITY_FEATURES +
    SESSION_FEATURES + REGIME_FEATURES + POSITION_FEATURES
)
