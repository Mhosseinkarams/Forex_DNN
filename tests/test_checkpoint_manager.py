import os
import shutil
import unittest
import json
import pandas as pd
from ML.checkpoint_manager import CheckpointManager

class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_checkpoints_run"
        os.makedirs(self.test_dir, exist_ok=True)
        self.symbol = "EURUSD"
        self.timeframe = "M5"
        self.window_size = 35
        self.stages = ["CLEAN", "ENRICH", "EVAL", "STATE", "LEVEL"]

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_mock_file(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        df.to_parquet(path)

    def test_checkpoint_initialization_and_save(self):
        """Verify that CheckpointManager initializes a fresh state and saves files on disk."""
        cp = CheckpointManager(
            symbol=self.symbol,
            timeframe=self.timeframe,
            window_size=self.window_size,
            stages=self.stages,
            checkpoint_dir=self.test_dir
        )
        self.assertEqual(cp.symbol, self.symbol)
        self.assertEqual(cp.get_next_unfinished_stage(), "CLEAN")
        self.assertTrue(os.path.exists(cp.checkpoint_path))
        self.assertTrue(os.path.exists(cp.backup_path))

    def test_stage_lifecycle_transitions(self):
        """Verify transitions through started and completed stage state changes."""
        cp = CheckpointManager(
            symbol=self.symbol,
            timeframe=self.timeframe,
            window_size=self.window_size,
            stages=self.stages,
            checkpoint_dir=self.test_dir
        )

        # Start stage CLEAN
        cp.mark_stage_started("CLEAN")
        self.assertEqual(cp.state["current_stage"], "CLEAN")
        self.assertTrue(cp.state["is_interrupted"])

        # Reload check: interrupted execution is detected
        cp_reload = CheckpointManager(
            symbol=self.symbol,
            timeframe=self.timeframe,
            window_size=self.window_size,
            stages=self.stages,
            checkpoint_dir=self.test_dir
        )
        self.assertTrue(cp_reload.state["is_interrupted"])
        self.assertEqual(cp_reload.get_next_unfinished_stage(), "CLEAN")

        # Complete stage CLEAN
        mock_output = os.path.join(self.test_dir, "clean_out.parquet")
        self._create_mock_file(mock_output)
        cp.mark_stage_completed("CLEAN", mock_output)

        self.assertIn("CLEAN", cp.state["completed_stages"])
        self.assertEqual(cp.get_stage_output("CLEAN"), mock_output)
        self.assertFalse(cp.state["is_interrupted"])
        self.assertEqual(cp.get_next_unfinished_stage(), "ENRICH")

    def test_reconciliation_on_missing_output(self):
        """Verify that completed stages are invalidated if their output files are deleted on disk."""
        cp = CheckpointManager(
            symbol=self.symbol,
            timeframe=self.timeframe,
            window_size=self.window_size,
            stages=self.stages,
            checkpoint_dir=self.test_dir
        )

        # Complete CLEAN and ENRICH with mock files on disk
        clean_out = os.path.join(self.test_dir, "clean.parquet")
        enrich_out = os.path.join(self.test_dir, "enrich.parquet")
        self._create_mock_file(clean_out)
        self._create_mock_file(enrich_out)

        cp.mark_stage_completed("CLEAN", clean_out)
        cp.mark_stage_completed("ENRICH", enrich_out)

        self.assertEqual(cp.get_next_unfinished_stage(), "EVAL")

        # Delete CLEAN's output file
        os.remove(clean_out)

        # Load fresh CheckpointManager context, triggering validation/reconciliation
        cp_new = CheckpointManager(
            symbol=self.symbol,
            timeframe=self.timeframe,
            window_size=self.window_size,
            stages=self.stages,
            checkpoint_dir=self.test_dir
        )

        # CLEAN's output was missing, so CLEAN and any subsequent stages are invalidated
        self.assertEqual(cp_new.get_next_unfinished_stage(), "CLEAN")
        self.assertNotIn("CLEAN", cp_new.state["completed_stages"])
        self.assertNotIn("ENRICH", cp_new.state["completed_stages"])

    def test_backup_restore_on_corruption(self):
        """Verify that if main checkpoint JSON is corrupted, CheckpointManager restores state from backup."""
        cp = CheckpointManager(
            symbol=self.symbol,
            timeframe=self.timeframe,
            window_size=self.window_size,
            stages=self.stages,
            checkpoint_dir=self.test_dir
        )

        # Complete CLEAN with valid output
        clean_out = os.path.join(self.test_dir, "clean.parquet")
        self._create_mock_file(clean_out)
        cp.mark_stage_completed("CLEAN", clean_out)

        self.assertEqual(cp.get_next_unfinished_stage(), "ENRICH")

        # Corrupt the main checkpoint file by writing invalid data to it
        with open(cp.checkpoint_path, "w", encoding="utf-8") as f:
            f.write("{ INVALID CORRUPTED JSON ")

        # Load CheckpointManager. It should fallback to the backup (.bak) file, recovering state successfully
        cp_recovered = CheckpointManager(
            symbol=self.symbol,
            timeframe=self.timeframe,
            window_size=self.window_size,
            stages=self.stages,
            checkpoint_dir=self.test_dir
        )

        self.assertIn("CLEAN", cp_recovered.state["completed_stages"])
        self.assertEqual(cp_recovered.get_next_unfinished_stage(), "ENRICH")
        # Ensure main checkpoint has been restored and is now valid again
        self.assertTrue(os.path.exists(cp_recovered.checkpoint_path))
        self.assertIsNotNone(cp_recovered._load_json(cp_recovered.checkpoint_path))

if __name__ == "__main__":
    unittest.main()
