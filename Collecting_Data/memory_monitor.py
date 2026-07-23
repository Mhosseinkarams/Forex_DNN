import os
import gc
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger("MemoryMonitor")

class MemoryMonitor:
    """
    Tracks and monitors RAM usage, object counts, and alerts when memory thresholds are exceeded.
    Provides an automatic cleanup strategy to free memory back to the OS.
    """
    def __init__(self, threshold_percent: float = 85.0, warning_limit_mb: float = 12000.0):
        self.threshold_percent = threshold_percent
        self.warning_limit_mb = warning_limit_mb
        try:
            self.process = psutil.Process(os.getpid())
        except Exception:
            self.process = None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Returns RAM usage statistics and object counts."""
        # Process memory
        rss_mb = 0.0
        vms_mb = 0.0
        if self.process:
            try:
                pmem_info = self.process.memory_info()
                rss_mb = pmem_info.rss / (1024 * 1024)
                vms_mb = pmem_info.vms / (1024 * 1024)
            except Exception:
                pass

        # System memory
        sys_percent = 0.0
        sys_available_mb = 0.0
        try:
            sys_mem = psutil.virtual_memory()
            sys_percent = sys_mem.percent
            sys_available_mb = sys_mem.available / (1024 * 1024)
        except Exception:
            pass

        # Object count monitoring
        object_count = 0
        df_count = 0
        arr_count = 0
        dict_count = 0
        try:
            all_objects = gc.get_objects()
            object_count = len(all_objects)

            # Count DataFrames, numpy arrays, and dicts safely
            for obj in all_objects:
                try:
                    tname = type(obj).__name__
                    if tname == 'DataFrame':
                        df_count += 1
                    elif tname == 'ndarray':
                        arr_count += 1
                    elif tname == 'dict':
                        dict_count += 1
                except Exception:
                    pass
        except Exception:
            pass

        return {
            "process_rss_mb": rss_mb,
            "process_vms_mb": vms_mb,
            "system_percent": sys_percent,
            "system_available_mb": sys_available_mb,
            "total_objects": object_count,
            "dataframe_count": df_count,
            "numpy_array_count": arr_count,
            "dict_count": dict_count
        }

    def check(self, context_msg: str = "") -> bool:
        """
        Checks current memory state. Logs stats, warns if threshold exceeded,
        and triggers automatic cleanup if necessary.
        Returns True if cleanup was triggered, False otherwise.
        """
        stats = self.get_memory_stats()
        rss = stats["process_rss_mb"]
        sys_pct = stats["system_percent"]

        logger.info(
            f"[Memory] Process: {rss:.1f} MB (VMS: {stats['process_vms_mb']:.1f} MB) | "
            f"System RAM: {sys_pct:.1f}% | Objects: {stats['total_objects']} "
            f"(DFs: {stats['dataframe_count']}, Arrays: {stats['numpy_array_count']}, Dicts: {stats['dict_count']}) "
            f"{' - ' + context_msg if context_msg else ''}"
        )

        exceeded = False
        if sys_pct >= self.threshold_percent or (rss > 0 and rss >= self.warning_limit_mb):
            exceeded = True
            logger.warning(
                f"[Memory Warning] Memory threshold exceeded! "
                f"System RAM: {sys_pct:.1f}% (Threshold: {self.threshold_percent:.1f}%) or "
                f"Process RSS: {rss:.1f} MB (Limit: {self.warning_limit_mb:.1f} MB)."
            )

        if exceeded:
            self.trigger_cleanup()
            return True
        return False

    def trigger_cleanup(self) -> Dict[str, Any]:
        """
        Executes automatic cleanup strategy:
        1. Clears FeaturePipeline feature caches
        2. Unreferences/clears any other known caches
        3. Forces explicit garbage collection
        """
        logger.info("[Memory] Triggering automatic memory cleanup...")
        before_stats = self.get_memory_stats()

        # 1. Clear known caches inside gc
        try:
            cleaned_pipelines = 0
            for obj in gc.get_objects():
                try:
                    if type(obj).__name__ == 'FeaturePipeline' and hasattr(obj, 'clear_cache'):
                        obj.clear_cache()
                        cleaned_pipelines += 1
                except Exception:
                    pass
            if cleaned_pipelines > 0:
                logger.info(f"[Memory] Cleared cache for {cleaned_pipelines} FeaturePipeline instances.")
        except Exception as e:
            logger.debug(f"[Memory] Error clearing FeaturePipeline caches during cleanup: {e}")

        # 2. Force explicit garbage collection
        gc.collect()
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

        after_stats = self.get_memory_stats()
        saved_mb = before_stats["process_rss_mb"] - after_stats["process_rss_mb"]
        saved_obj = before_stats["total_objects"] - after_stats["total_objects"]

        logger.info(
            f"[Memory] Cleanup complete. Process RSS: {before_stats['process_rss_mb']:.1f} MB -> {after_stats['process_rss_mb']:.1f} MB "
            f"(Saved: {saved_mb:.1f} MB). Objects: {before_stats['total_objects']} -> {after_stats['total_objects']} (Saved: {saved_obj})."
        )
        return {
            "saved_mb": saved_mb,
            "saved_objects": saved_obj
        }
