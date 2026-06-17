import pandas as pd
import numpy as np
from .indicators import IndicatorEngine

class TechnicalIndicators:
    """
    A class to calculate technical indicators for Forex data.
    Refactored to use IndicatorEngine for Module 2 consistency.
    """

    @staticmethod
    def add_all_indicators(df):
        """
        Adds all available technical indicators to the dataframe.
        Wraps IndicatorEngine to maintain backward compatibility.
        """
        # Ensure numeric types and handle potential string artifacts from yfinance MultiIndex headers
        # We perform basic cleaning before passing to IndicatorEngine
        df = df.copy()
        for col in ['Open', 'High', 'Low', 'Close', 'Vol', 'Volume', 'TickVolume', 'Spread']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Map 'Vol' or 'Volume' to 'TickVolume' if missing, as IndicatorEngine expects 'TickVolume'
        if 'TickVolume' not in df.columns:
            if 'Vol' in df.columns:
                df['TickVolume'] = df['Vol']
            elif 'Volume' in df.columns:
                df['TickVolume'] = df['Volume']
            else:
                df['TickVolume'] = 0

        if 'Spread' not in df.columns:
            df['Spread'] = 0

        # Drop rows where essential price data is missing
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

        engine = IndicatorEngine(dropna=True)
        df_enriched = engine.calculate(df)

        # For legacy compatibility, map some columns back to previous names if they differ
        # (Though most scripts should transition to the new names)
        # Old names: EMA_50, EMA_200, ATR, Candle_Body, Upper_Shadow, Lower_Shadow
        if 'ema_50' in df_enriched.columns:
            df_enriched['EMA_50'] = df_enriched['ema_50']

        # EMA_200 was in old utils but not in new IndicatorEngine defaults.
        # We can add it manually or add it to ema_periods.
        if 'EMA_200' not in df_enriched.columns:
            df_enriched['EMA_200'] = df_enriched['Close'].ewm(span=200, adjust=False).mean()

        if 'atr_14' in df_enriched.columns:
            df_enriched['ATR'] = df_enriched['atr_14']

        if 'body_size' in df_enriched.columns:
            df_enriched['Candle_Body'] = df_enriched['body_size']

        if 'upper_shadow' in df_enriched.columns:
            df_enriched['Upper_Shadow'] = df_enriched['upper_shadow']

        if 'lower_shadow' in df_enriched.columns:
            df_enriched['Lower_Shadow'] = df_enriched['lower_shadow']

        # Legacy RSI and MACD (not in new IndicatorEngine but might be used by existing scripts)
        delta = df_enriched['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df_enriched['RSI'] = 100 - (100 / (1 + rs))

        exp1 = df_enriched['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_enriched['Close'].ewm(span=26, adjust=False).mean()
        df_enriched['MACD'] = exp1 - exp2
        df_enriched['MACD_Signal'] = df_enriched['MACD'].ewm(span=9, adjust=False).mean()

        return df_enriched
