import pandas as pd
import numpy as np
import logging

# Configure logger
logger = logging.getLogger('IndicatorEngine')

class IndicatorEngine:
    def __init__(
        self,
        ema_periods: list[int] = [21, 50, 600],
        atr_period: int = 14,
        body_avg_window: int = 20,
        shadow_ratio_window: int = 5,
        slope_period: int = 32,
        dropna: bool = False,
    ):
        self.ema_periods = ema_periods
        self.atr_period = atr_period
        self.body_avg_window = body_avg_window
        self.shadow_ratio_window = shadow_ratio_window
        self.slope_period = slope_period
        self.dropna = dropna

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Receives a DataFrame in standard schema.
        Returns a new DataFrame (never modifies input) with all
        Layer 1 and Layer 2 columns appended.
        If dropna=False (default): NaN rows are preserved.
            Log a warning if any EMA or ATR column contains NaN.
        If dropna=True: drop rows where any EMA or ATR column
            contains NaN. Use for backtesting only.
        """
        # 1. Never modify the input DataFrame. Work on a copy.
        original_cols = list(df.columns)
        df = df.copy()

        if len(df) < max(self.ema_periods) + self.slope_period:
            logger.warning(f"Input DataFrame has fewer rows ({len(df)}) than warmup requirement ({max(self.ema_periods) + self.slope_period}). Indicators may not be fully warmed up.")

        # Layer 1 — Indicators
        layer1_cols = []
        for p in self.ema_periods:
            col_name = f'ema_{p}'
            df[col_name] = df['Close'].ewm(span=p, adjust=False).mean()
            layer1_cols.append(col_name)

        # ATR: Wilder's method
        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr_col = f'atr_{self.atr_period}'
        df[atr_col] = true_range.ewm(alpha=1/self.atr_period, adjust=False).mean()
        layer1_cols.append(atr_col)

        # Layer 2 — Candle Metadata
        layer2_cols = []
        
        df['body_size'] = (df['Close'] - df['Open']).abs()
        layer2_cols.append('body_size')
        
        df['avg_body_size'] = df['body_size'].rolling(window=self.body_avg_window, min_periods=1).mean()
        layer2_cols.append('avg_body_size')
        
        # body_ratio: use 0.0 where avg is 0
        df['body_vs_avg'] = np.where(df['avg_body_size'] != 0, df['body_size'] / df['avg_body_size'], 0.0)
        layer2_cols.append('body_vs_avg')

        df['candle_size']=abs(df['High'] - df['Low'])
        layer2_cols.append('candle_size')
        
        df['body_pct'] = np.where(df['candle_size'] != 0, df['body_size'] / df['candle_size'], 0.0)
        layer2_cols.append('body_pct')
        
        df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        layer2_cols.append('upper_shadow')
        
        df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        layer2_cols.append('lower_shadow')
        
        df['total_shadow'] = df['upper_shadow'] + df['lower_shadow']
        layer2_cols.append('total_shadow')
        
        df['body_shadow_ratio'] = df['body_size'] / (df['total_shadow'] + 1e-9)
        layer2_cols.append('body_shadow_ratio')
        
        df['rolling_body_shadow_ratio'] = df['body_shadow_ratio'].rolling(window=self.shadow_ratio_window, min_periods=1).mean()
        layer2_cols.append('rolling_body_shadow_ratio')
        
        df['candle_direction'] = np.where(df['Close'] > df['Open'], 1, np.where(df['Close'] < df['Open'], -1, 0))
        layer2_cols.append('candle_direction')

        # EMA-relative features
        for p in self.ema_periods:
            ema_col = f'ema_{p}'
            
            dist_col = f'dist_ema_{p}'
            df[dist_col] = (df['Close'] - df[ema_col]) / df[atr_col]
            layer2_cols.append(dist_col)
            
            cross_col = f'cross_ema_{p}'
            prev_close_val = df['Close'].shift(1)
            prev_ema_val = df[ema_col].shift(1)
            df[cross_col] = 0
            df.loc[(prev_close_val < prev_ema_val) & (df['Close'] > df[ema_col]), cross_col] = 1
            df.loc[(prev_close_val > prev_ema_val) & (df['Close'] < df[ema_col]), cross_col] = -1
            layer2_cols.append(cross_col)
            
            span_col = f'ema_span_{p}'
            df[span_col] = np.where((df['Low'] <= df[ema_col]) & (df[ema_col] <= df['High']), 1, 0)
            layer2_cols.append(span_col)

        # EMA slope — EMA 600 only
        # slope_period=32 on EMA 600 means measuring EMA displacement 
        # over 160 minutes (~2.5 hours) relative to a 3000-minute EMA. 
        # This is 32/600 ≈ 5% of the EMA period. Configurable for 
        # optimization. Slope applies to EMA 600 (MM strategy) only.
        if 600 in self.ema_periods:
            df['ema_slope_600'] = (df['ema_600'] - df['ema_600'].shift(self.slope_period)).abs() / df[atr_col]
            layer2_cols.append('ema_slope_600')

        # Handle NaN values and logging
        if self.dropna:
            initial_len = len(df)
            df.dropna(subset=layer1_cols, inplace=True)
            dropped_count = initial_len - len(df)
            logger.info(f"Dropped {dropped_count} rows containing NaN in EMA or ATR columns.")
        else:
            if df[layer1_cols].isnull().any().any():
                logger.warning("NaN values detected in EMA or ATR columns.")

        # Reorder columns
        final_cols = original_cols + layer1_cols + layer2_cols
        return df[final_cols]

if __name__ == "__main__": 
    import pandas as pd 
    import numpy as np 
 
    # Set up basic logging to see output
    logging.basicConfig(level=logging.INFO)

    # Synthetic test: verifies columns are produced and 
    # types are correct. Not a strategy test. 
    n = 700  # sufficient to warm EMA 600 
    np.random.seed(42) 
    prices = 1.1000 + np.cumsum(np.random.randn(n) * 0.0001) 
 
    df_test = pd.DataFrame({ 
        "Datetime":   pd.date_range("2024-01-01", periods=n, freq="5min"), 
        "Open":       prices + np.random.randn(n) * 0.0001, 
        "High":       prices + np.abs(np.random.randn(n)) * 0.0003, 
        "Low":        prices - np.abs(np.random.randn(n)) * 0.0003, 
        "Close":      prices, 
        "TickVolume": np.random.randint(100, 1000, n), 
        "Spread":     np.zeros(n, dtype=int), 
    }) 
 
    engine = IndicatorEngine() 
    result = engine.calculate(df_test) 
 
    print(f"Input rows:  {len(df_test)}") 
    print(f"Output rows: {len(result)}") 
    print(f"Columns ({len(result.columns)}): {list(result.columns)}") 
    print(result.tail(3).to_string()) 
 
    # Assertions 
    assert len(result) == len(df_test), "Row count must not change when dropna=False" 
    assert "ema_600" in result.columns 
    assert "dist_ema_600" in result.columns 
    assert "ema_slope_600" in result.columns 
    assert "ema_slope_21" not in result.columns, "Slope is EMA 600 only" 
    assert "ema_slope_50" not in result.columns, "Slope is EMA 600 only" 
    assert result["cross_ema_21"].isin([-1, 0, 1]).all() 
    assert result["candle_direction"].isin([-1, 0, 1]).all() 
    
    # Check column order
    expected_start = ["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"]
    assert list(result.columns[:7]) == expected_start, "Original columns order incorrect"
    
    print("All assertions passed.")
