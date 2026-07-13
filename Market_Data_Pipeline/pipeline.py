import logging
from typing import List, Dict, Any, Optional, Union
import pandas as pd

logger = logging.getLogger("Pipeline")

class PipelineContext:
    """Carries shared data and results between pipeline stages."""
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data: Dict[str, Any] = {}

class Pipeline:
    """
    Configurable Pipeline for executing analysis engines and feature extraction sequentially.
    Allows future Smart Money / technical analysis engines to be registered as plug-ins.
    """
    def __init__(self):
        self.stages: List[Any] = []

    def register(self, stage: Any) -> "Pipeline":
        """Registers a stage to the pipeline."""
        self.stages.append(stage)
        logger.info(f"Registered pipeline stage: {stage.__class__.__name__}")
        return self

    def execute(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Executes all registered whole-dataframe transformer/analysis stages.
        Updates df sequentially.
        """
        df_transformed = df.copy()

        # Ensure Datetime column is datetime type
        if "Datetime" in df_transformed.columns:
            df_transformed["Datetime"] = pd.to_datetime(df_transformed["Datetime"])

        for stage in self.stages:
            stage_name = stage.__class__.__name__
            # Skip registry and label_engine / feature pipeline for whole-dataframe transformation
            if stage_name in ("FeatureRegistry", "LabelEngine", "FeaturePipeline"):
                continue

            try:
                if hasattr(stage, "calculate"):
                    logger.debug(f"Executing {stage_name}.calculate for {symbol}...")
                    df_transformed = stage.calculate(df_transformed)
                elif hasattr(stage, "process"):
                    logger.debug(f"Executing {stage_name}.process for {symbol}...")
                    df_transformed = stage.process(df_transformed)
                else:
                    # Generic stage call if support exists
                    if callable(stage):
                        logger.debug(f"Executing callable {stage_name} for {symbol}...")
                        df_transformed = stage(df_transformed)
            except Exception as e:
                logger.error(f"Error executing pipeline stage {stage_name} on {symbol}: {e}", exc_info=True)
                raise e

        return df_transformed

    def get_stage(self, stage_class: Any) -> Optional[Any]:
        """Retrieves a registered stage instance by class or name."""
        for stage in self.stages:
            if isinstance(stage_class, str):
                if stage.__class__.__name__ == stage_class:
                    return stage
            else:
                if isinstance(stage, stage_class):
                    return stage
        return None

class PipelineRegistry:
    """Registry managing standard / custom pre-configured pipelines."""
    _pipelines: Dict[str, Pipeline] = {}

    @classmethod
    def register_pipeline(cls, name: str, pipeline: Pipeline):
        cls._pipelines[name] = pipeline

    @classmethod
    def get_pipeline(cls, name: str) -> Optional[Pipeline]:
        return cls._pipelines.get(name)
