import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    A class to calculate technical indicators for Forex data.
    """

    @staticmethod
    def add_all_indicators(df):
        """
        Adds all available technical indicators to the dataframe.
        """
        df = df.copy()

        # Ensure numeric types and handle potential string artifacts from yfinance MultiIndex headers
        for col in ['Open', 'High', 'Low', 'Close', 'Vol', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where essential price data is missing or corrupted (like ticker strings)
        df.dropna(subset=['Close'], inplace=True)

        # EMA
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # ATR (14)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Candle and Shadow Sizes
        df['Candle_Body'] = np.abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']

        # Ichimoku Cloud
        period9_high = df['High'].rolling(window=9).max()
        period9_low = df['Low'].rolling(window=9).min()
        df['Tenkan_Sen'] = (period9_high + period9_low) / 2

        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()
        df['Kijun_Sen'] = (period26_high + period26_low) / 2

        df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)

        period52_high = df['High'].rolling(window=52).max()
        period52_low = df['Low'].rolling(window=52).min()
        df['Senkou_Span_B'] = ((period52_high + period52_low) / 2).shift(26)

        df['Chikou_Span_Rel'] = df['Close'] - df['Close'].shift(26)

        df.dropna(inplace=True)
        return df
