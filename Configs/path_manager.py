import os

class PathManager:
    """
    Centralized path manager that acts as the single source of truth for all directories.
    All modules request directories from it instead of building paths manually.
    """
    # Base directory is the root of the repository (Forex_DNN/)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Configured locations mapped under the BASE_DIR
    PATHS = {
        "historical_data": os.path.join(BASE_DIR, "Data", "Historical"),
        "processed_data": os.path.join(BASE_DIR, "Data", "ML", "Processed"),
        "feature_data": os.path.join(BASE_DIR, "Data", "ML", "Features"),
        "labels": os.path.join(BASE_DIR, "Data", "ML", "Labels"),
        "datasets": os.path.join(BASE_DIR, "Data", "ML", "Datasets"),
        "models": os.path.join(BASE_DIR, "Data", "ML", "Models"),
        "reports": os.path.join(BASE_DIR, "Data", "ML", "Reports"),
        "cache": os.path.join(BASE_DIR, "Data", "Cache"),
        "temporary": os.path.join(BASE_DIR, "Data", "Temporary"),
        "logs": os.path.join(BASE_DIR, "Logs"),
        "journals": os.path.join(BASE_DIR, "Journals"),
        "config": os.path.join(BASE_DIR, "Configs"),
        "documentation": os.path.join(BASE_DIR, "Docs"),
    }

    @classmethod
    def get_path(cls, key: str, *subpaths: str) -> str:
        """
        Get the absolute path for a given location key and optionally append subpaths.
        """
        if key not in cls.PATHS:
            raise KeyError(f"Unknown path key: '{key}'. Available keys: {list(cls.PATHS.keys())}")

        path = cls.PATHS[key]
        if subpaths:
            path = os.path.join(path, *subpaths)
        return path

    @classmethod
    def get_relative_path(cls, key: str, *subpaths: str) -> str:
        """
        Get a path relative to the current working directory.
        """
        abs_path = cls.get_path(key, *subpaths)
        return os.path.relpath(abs_path, os.getcwd())

    @classmethod
    def ensure_all_dirs(cls):
        """
        Ensures all top-level and sub-directories exist.
        """
        for key, path in cls.PATHS.items():
            os.makedirs(path, exist_ok=True)
