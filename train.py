#!/usr/bin/env python3
"""
train.py

Unified ML training pipeline orchestrator for the Forex_DNN framework.
This script acts purely as an orchestrator, parsing command line options and
delegating execution to the modular Pipeline.training_pipeline engine.
It contains absolutely zero model-specific training logic.

Usage:
    # Train all enabled models in optimal dependency order
    python train.py --all

    # Train a specific model
    python train.py --model market_state

    # Train with a custom configuration
    python train.py --config Configs/training_config.yaml
"""

import sys
import argparse
import logging

from Pipeline.training_pipeline import TrainingPipeline

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("Logs/train_pipeline.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("TrainOrchestrator")


def main():
    parser = argparse.ArgumentParser(description="Forex_DNN Unified ML Training Pipeline Orchestrator")
    parser.add_argument("--all", action="store_true",
                        help="Train all enabled model configurations in dependency order.")
    parser.add_argument("--model", type=str, default=None,
                        choices=["market_state", "level_break"],
                        help="Specific model to train.")
    parser.add_argument("--config", type=str, default="Configs/training_config.yaml",
                        help="Path to YAML configuration containing training parameters.")
    parser.add_argument("--force", action="store_true",
                        help="Force pipeline execution to continue even if an individual trainer fails.")

    args = parser.parse_args()

    # Integrity Check: user must specify either --all or a specific --model
    if not args.all and not args.model:
        parser.print_help()
        print("\nError: You must specify either --all to train all models or --model <name> to train a specific model.")
        sys.exit(1)

    try:
        # Initialize training pipeline
        pipeline = TrainingPipeline(config_path=args.config)

        # Run pipeline
        target_model = "all" if args.all else args.model
        success = pipeline.run_all(model_name=target_model, force=args.force)

        if not success:
            logger.error("Training pipeline run failed!")
            sys.exit(1)

        logger.info("Training pipeline run successfully finalized!")

    except Exception as e:
        logger.error(f"Uncaught exception during training pipeline orchestration: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
