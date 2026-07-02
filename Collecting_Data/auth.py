import os
import json
from pathlib import Path
from dotenv import load_dotenv

def load_credentials(path="credentials.json"):
    """
    Loads MT5 credentials from credentials.json or environment variables.
    Returns a dictionary with 'login', 'password', and 'server'.
    """
    # Default values from environment
    load_dotenv()
    creds = {
        "login": int(os.getenv("MT5_ID", 0)),
        "password": os.getenv("MT5_PASSWORD", ""),
        "server": os.getenv("MT5_SERVER", "")
    }

    # Try to load from credentials.json
    json_path = Path(path)
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                mt5_data = data.get("mt5", {})
                creds["login"] = int(mt5_data.get("login", creds["login"]))
                creds["password"] = mt5_data.get("password", creds["password"])
                creds["server"] = mt5_data.get("server", creds["server"])
                
        except Exception as e:
            print(f"Warning: Failed to load credentials.json: {e}")
    return creds
