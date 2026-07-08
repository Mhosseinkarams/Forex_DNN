import os
import json
from pathlib import Path
from dotenv import load_dotenv

def load_credentials(path="credentials.json"):
    """
    Purpose:
        Loads MetaTrader 5 credentials from a JSON file or environment variables.
        Provides a centralized way for all modules to authenticate with the broker.

    Arguments:
        path (str): Path to the JSON file containing credentials. Defaults to "credentials.json".

    Returns:
        dict: A dictionary containing 'login' (int), 'password' (str), and 'server' (str).

    Exceptions:
        Logs a warning if the JSON file is missing or malformed, falling back to environment variables.

    Example:
        >>> creds = load_credentials("credentials.json")
        >>> print(creds['login'])
        12345678

    Notes:
        Prioritizes: credentials.json > .env file > environment variables.
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
