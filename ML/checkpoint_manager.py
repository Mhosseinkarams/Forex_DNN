import os
import json
import logging
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("CheckpointManager")

class CheckpointManager:
    """
    Resilient and production-quality Checkpoint Manager for the Forex_DNN framework.
    Persists stage-based execution states on disk, enabling automatic resume and crash recovery.
    Uses atomic writes with temporary swap, tracks interrupted executions, and maintains
    a backup (.bak) file to guarantee immunity from file corruption.
    """
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        window_size: int,
        stages: List[str],
        checkpoint_dir: str,
        pipeline_version: str = "2.0.0"
    ):
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.window_size = window_size
        self.stages = stages
        self.checkpoint_dir = checkpoint_dir
        self.pipeline_version = pipeline_version

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.symbol}_{self.timeframe}_w{self.window_size}_checkpoint.json")
        self.backup_path = self.checkpoint_path + ".bak"

        # Initialize or load state
        self.state = self._load_or_initialize()

    def _initialize_empty_state(self) -> Dict[str, Any]:
        """Creates a fresh, default state dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "window_size": self.window_size,
            "pipeline_version": self.pipeline_version,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "completed_stages": [],
            "current_stage": None,
            "stage_outputs": {},
            "is_interrupted": False
        }

    def _load_or_initialize(self) -> Dict[str, Any]:
        """
        Attempts to load checkpoint from the main state file.
        If corrupted or missing, falls back to the backup (.bak) file.
        If both are missing or unreadable, initializes a fresh state.
        """
        # Try loading main checkpoint
        if os.path.exists(self.checkpoint_path):
            state = self._load_json(self.checkpoint_path)
            if state is not None:
                if self._validate_and_reconcile_state(state):
                    logger.info(f"[{self.symbol}] Successfully loaded valid checkpoint from disk.")
                    return state
                else:
                    logger.warning(f"[{self.symbol}] Main checkpoint failed validation. Checking backup...")
            else:
                logger.warning(f"[{self.symbol}] Main checkpoint was corrupted. Checking backup...")

        # Try loading backup checkpoint
        if os.path.exists(self.backup_path):
            state = self._load_json(self.backup_path)
            if state is not None:
                if self._validate_and_reconcile_state(state):
                    logger.info(f"[{self.symbol}] Recovered valid checkpoint state from backup.")
                    # Sync backup back to main
                    self._save_state_atomically(state)
                    return state
                else:
                    logger.warning(f"[{self.symbol}] Backup checkpoint failed validation.")
            else:
                logger.warning(f"[{self.symbol}] Backup checkpoint was also corrupted.")

        # Initialize fresh state if both failed
        logger.info(f"[{self.symbol}] No valid checkpoint found. Initializing fresh processing state.")
        state = self._initialize_empty_state()
        self._save_state_atomically(state)
        return state

    def _load_json(self, path: str) -> Optional[Dict[str, Any]]:
        """Safely loads and parses a JSON file, returning None on failure."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read/parse checkpoint file {path}: {e}")
            return None

    def _validate_and_reconcile_state(self, state: Dict[str, Any]) -> bool:
        """
        Validates the structure of the loaded state.
        Reconciles completed stages by ensuring their output files actually exist on disk.
        If an intermediate file is missing, truncates completed stages back to that point.
        """
        try:
            # Check basic structure
            required_keys = ["symbol", "timeframe", "window_size", "pipeline_version", "completed_stages", "stage_outputs"]
            for k in required_keys:
                if k not in state:
                    logger.error(f"Missing required key '{k}' in checkpoint state.")
                    return False

            # Verify context metadata
            if state["symbol"].upper() != self.symbol or state["timeframe"] != self.timeframe or state["window_size"] != self.window_size:
                logger.warning(f"Checkpoint context mismatch. Expected: {self.symbol}/{self.timeframe}/w{self.window_size}. Got: {state['symbol']}/{state['timeframe']}/w{state['window_size']}")
                return False

            # Reconcile completed stages with actual disk files
            valid_completed = []
            valid_outputs = {}

            for stage in self.stages:
                if stage in state["completed_stages"]:
                    out_path = state["stage_outputs"].get(stage)
                    if out_path and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                        valid_completed.append(stage)
                        valid_outputs[stage] = out_path
                    else:
                        logger.warning(f"[{self.symbol}] Output for stage '{stage}' is missing or empty at '{out_path}'. Invalidation triggered.")
                        # Stop importing any further stages if an intermediate stage's output is missing
                        break

            state["completed_stages"] = valid_completed
            state["stage_outputs"] = valid_outputs

            # If current_stage was left non-empty on load, it means previous run crashed/interrupted mid-execution
            if state.get("current_stage") is not None:
                state["is_interrupted"] = True
                logger.info(f"[{self.symbol}] Interrupted execution detected. Left in stage '{state['current_stage']}'.")

            return True
        except Exception as e:
            logger.error(f"Error validating checkpoint state: {e}")
            return False

    def _save_state_atomically(self, state: Dict[str, Any]) -> None:
        """
        Writes the state dictionary to disk atomically.
        Writes to a temporary (.tmp) file, syncs/closes, and then os.replace-s it.
        Also duplicates to a backup (.bak) file to prevent any corruption.
        """
        temp_path = self.checkpoint_path + ".tmp"
        try:
            # Write atomically to temp
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
                f.flush()
                os.fsync(f.fileno())

            # Atomically replace main file
            os.replace(temp_path, self.checkpoint_path)

            # Duplicate to backup file atomically as well
            temp_bak_path = self.backup_path + ".tmp"
            with open(temp_bak_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_bak_path, self.backup_path)

        except Exception as e:
            logger.error(f"Failed to write checkpoint state atomically for {self.symbol}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def get_next_unfinished_stage(self) -> Optional[str]:
        """
        Finds and returns the first stage in our stages sequence that has not completed.
        Returns None if all stages are completed.
        """
        for stage in self.stages:
            if stage not in self.state["completed_stages"]:
                return stage
        return None

    def mark_stage_started(self, stage_name: str) -> None:
        """Sets the stage as current/running and persists state to disk."""
        if stage_name not in self.stages:
            raise ValueError(f"Invalid stage name: {stage_name}")

        self.state["current_stage"] = stage_name
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.state["is_interrupted"] = True
        self._save_state_atomically(self.state)
        logger.info(f"[{self.symbol}] Transitioning state: Stage '{stage_name}' started.")

    def mark_stage_completed(self, stage_name: str, output_path: str) -> None:
        """
        Appends the stage to completed_stages, records its output path,
        clears current_stage, and persists state to disk.
        """
        if stage_name not in self.stages:
            raise ValueError(f"Invalid stage name: {stage_name}")

        if stage_name not in self.state["completed_stages"]:
            self.state["completed_stages"].append(stage_name)
        self.state["stage_outputs"][stage_name] = output_path
        self.state["current_stage"] = None
        self.state["is_interrupted"] = False
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._save_state_atomically(self.state)
        logger.info(f"[{self.symbol}] Transitioning state: Stage '{stage_name}' successfully completed.")

    def get_stage_output(self, stage_name: str) -> str:
        """Returns the registered output path for a completed stage."""
        if stage_name not in self.state["completed_stages"]:
            raise ValueError(f"Stage '{stage_name}' is not completed yet!")
        return self.state["stage_outputs"][stage_name]

    def reset(self) -> None:
        """Resets the state entirely, wiping completed/current stages and output files."""
        # Optional: delete actual file outputs
        for stage, path in list(self.state["stage_outputs"].items()):
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"[{self.symbol}] Deleted stage output on reset: {path}")
                except Exception as e:
                    logger.warning(f"Could not delete stage output {path}: {e}")

        self.state = self._initialize_empty_state()
        self._save_state_atomically(self.state)
        logger.info(f"[{self.symbol}] Checkpoint state has been reset.")
