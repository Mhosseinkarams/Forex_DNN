import os
import time
import logging

logger = logging.getLogger("Utils")

def safe_file_replace(src: str, dst: str, max_retries: int = 5, delay: float = 0.1):
    """
    Purpose:
        Atomically replaces a file with another, implementing a retry mechanism
        to handle intermittent 'Access is denied' (WinError 5) errors on Windows.

    Arguments:
        src (str): Source file path (the temporary file).
        dst (str): Destination file path (the final state file).
        max_retries (int): Maximum number of attempts.
        delay (float): Seconds to wait between retries.

    Returns:
        bool: True if successful, False otherwise.
    """
    for i in range(max_retries):
        try:
            if os.path.exists(dst):
                # On Windows, os.replace might fail if the file is being read
                # even if it's supposed to be an atomic operation.
                os.replace(src, dst)
            else:
                os.rename(src, dst)
            return True
        except OSError as e:
            if i < max_retries - 1:
                logger.warning(f"Retry {i+1}/{max_retries} replacing {dst} due to: {e}")
                time.sleep(delay)
            else:
                logger.error(f"Failed to replace {dst} after {max_retries} attempts: {e}")
                return False
    return False
