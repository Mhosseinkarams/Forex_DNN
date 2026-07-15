#!/usr/bin/env python3
"""
trade.py

Unified trading entry point for the Forex_DNN framework.
Serves as the sole entry point to bootstrap live, demo, paper, validation,
backtest, or simulation runs. Wires all components seamlessly based on configuration.

Usage:
    # Run backtesting using Configs/trading_config.yaml settings
    python trade.py --mode backtest

    # Run live trading on a custom symbol list
    python trade.py --mode live --symbols EURUSD

    # Run in ML Shadow Mode
    python trade.py --mode backtest --shadow
"""

import sys
import argparse
import logging

from Pipeline.trading_pipeline import TradingPipeline

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("Logs/trade_pipeline.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("TradeOrchestrator")


def main():
    parser = argparse.ArgumentParser(description="Forex_DNN Unified Trading Entry Point")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["live", "demo", "paper", "validation", "backtest", "simulation"],
                        help="The execution run mode (Live, Demo, Paper, Validation, Backtest, Simulation).")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbol list to trade (e.g. EURUSD,GBPUSD).")
    parser.add_argument("--config", type=str, default="Configs/trading_config.yaml",
                        help="Path to YAML configuration containing pipeline parameters.")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["mm_strategy", "sm_strategy", "unit_strategy"],
                        help="Specific trading strategy to activate.")
    parser.add_argument("--shadow", action="store_true",
                        help="Activate machine learning shadow monitoring mode.")

    args = parser.parse_args()

    try:
        # 1. Initialize Pipeline with configuration file
        pipeline = TradingPipeline(config_path=args.config)

        # 2. Inject command line argument overrides
        if args.mode:
            pipeline.config["trading_mode"] = args.mode
        if args.symbols:
            pipeline.config["symbols"] = [s.strip().upper() for s in args.symbols.split(",")]
        if args.shadow:
            pipeline.config["shadow_mode"] = True

        if args.strategy:
            # Disable all other strategies, enable the requested one
            for strat_name in pipeline.config.get("strategies", {}):
                pipeline.config["strategies"][strat_name]["enabled"] = (strat_name == args.strategy)

        # 3. Bootstrap pipeline (wires sizers, tracking, journals, exits, strategy)
        if not pipeline.bootstrap():
            logger.error("Failed to bootstrap trading pipeline! Terminating.")
            sys.exit(1)

        logger.info("Trading Pipeline successfully bootstrapped!")

        # 4. Launch execution loop or background threads
        pipeline.run()

    except Exception as e:
        logger.error(f"Uncaught exception during trading pipeline orchestration: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
