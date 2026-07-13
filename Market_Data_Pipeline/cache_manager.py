import os
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger("DatasetCacheManager")

class DatasetCacheManager:
    """
    Handles caching and resume for the HistoricalDatasetBuilder.
    Caches computed features/labels per symbol and resumes automatically if interrupted.
    """
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, symbol: str, timeframe: str, version: str) -> str:
        # File path for cache
        return os.path.join(self.cache_dir, f"{symbol}_{timeframe}_{version}_cache.parquet")

    def get_cached_symbol(self, symbol: str, timeframe: str, version: str) -> Optional[pd.DataFrame]:
        """Loads cached DataFrame for the symbol if it exists."""
        cache_path = self._get_cache_path(symbol, timeframe, version)
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                logger.info(f"Loaded cached data for {symbol} ({len(df)} rows) from {cache_path}")
                return df
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_path}: {e}")
        return None

    def cache_symbol(self, symbol: str, timeframe: str, version: str, df: pd.DataFrame) -> None:
        """Saves processed symbol DataFrame to cache."""
        cache_path = self._get_cache_path(symbol, timeframe, version)
        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached processed data for {symbol} to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache for {symbol} at {cache_path}: {e}")

    def clear_cache(self) -> None:
        """Clears all cache files in cache_dir."""
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith(".parquet"):
                    try:
                        os.remove(os.path.join(self.cache_dir, file))
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {file}: {e}")
            logger.info("Cleared cache directory.")
