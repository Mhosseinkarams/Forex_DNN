import json
import os

class DebugConfig:
    """
    Configuration helper for Visualization and Interactive Debugging layers.
    Allows toggling layers on and off dynamically.
    """
    def __init__(self, config_file: str = "debug_config.json"):
        self.config_file = config_file
        self.layers = {
            "swings": True,
            "structure": True,
            "zones": True,
            "levels": True,
            "signals": True,
            "ml": True
        }
        self.load()

    def load(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    self.layers.update(data.get("layers", {}))
            except Exception:
                pass

    def save(self):
        try:
            with open(self.config_file, "w") as f:
                json.dump({"layers": self.layers}, f, indent=4)
        except Exception:
            pass

    def set_layer(self, layer_name: str, enabled: bool):
        if layer_name in self.layers:
            self.layers[layer_name] = enabled
            self.save()

    def is_enabled(self, layer_name: str) -> bool:
        return self.layers.get(layer_name, True)
