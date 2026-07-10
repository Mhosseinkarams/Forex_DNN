import os
import csv
from typing import List, Dict
import logging
from Visualization.render_types import DrawInstruction

logger = logging.getLogger("DrawInstructionWriter")

class DrawInstructionWriter:
    """
    Purpose:
        Responsible for exporting lists of DrawInstruction objects into highly structured,
        independent CSV files (one per category/layer per symbol) inside the MT5 Files directory
        or any target output directory.
    """
    @staticmethod
    def write_instructions(filepath: str, instructions: List[DrawInstruction]):
        """
        Write a list of DrawInstruction objects to a CSV file.
        Uses an atomic write pattern (temp file then rename) to prevent MT5 from reading partial/corrupt files.
        """
        temp_filepath = filepath + ".tmp"
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(temp_filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(["TYPE", "NAME", "TIME1", "TIME2", "PRICE1", "PRICE2", "COLOR", "STYLE", "TEXT"])
                # Write data rows
                for inst in instructions:
                    writer.writerow([
                        inst.type_name,
                        inst.name,
                        inst.time1,
                        inst.time2,
                        inst.price1,
                        inst.price2,
                        inst.color,
                        inst.style,
                        inst.text
                    ])

            # Atomic swap
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_filepath, filepath)
            logger.debug(f"Successfully wrote {len(instructions)} instructions to {filepath}")
        except Exception as e:
            logger.error(f"Failed to write draw instructions to {filepath}: {e}")
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception:
                    pass
