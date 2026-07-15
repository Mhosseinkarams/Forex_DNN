import logging
from typing import Dict, List, Any, Optional, Tuple

# Import standard implementations from pipeline to avoid duplication
from Market_Data_Pipeline.refusal_candle_engine import (
    RefusalCandleEngine as PipelineRefusalCandleEngine,
    RefusalSignal as PipelineRefusalSignal
)

logger = logging.getLogger("LegacyRefusalCandleEngineWrapper")

# Provide backward compatible aliases
RefusalResult = PipelineRefusalSignal

class RefusalCandleEngine(PipelineRefusalCandleEngine):
    """
    Backward-compatible wrapper for Strategies/refusal_candle_engine.py.
    Inherits everything from the centralized pipeline engine to prevent logic duplication.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
