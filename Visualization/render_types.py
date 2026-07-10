from dataclasses import dataclass
from typing import Optional

@dataclass
class DrawInstruction:
    type_name: str     # TYPE (e.g., SWING, BOS, CHOCH, ZONE, LEVEL, TEXT, PANEL)
    name: str          # NAME (unique object ID)
    time1: str = ""    # TIME1 (YYYY.MM.DD HH:MM:SS or empty)
    time2: str = ""    # TIME2 (YYYY.MM.DD HH:MM:SS or empty)
    price1: str = ""   # PRICE1 (float or empty)
    price2: str = ""   # PRICE2 (float or empty)
    color: str = ""    # COLOR (e.g., Red, Blue, Green, Gray, Orange, Cyan, etc.)
    style: str = ""    # STYLE (e.g., Solid, Dash, Dot, etc.)
    text: str = ""     # TEXT (any label or annotation text)

    def to_csv_row(self) -> str:
        # Escape commas in text/name to avoid splitting issues in CSV
        escaped_text = self.text.replace(",", ";").replace("\n", " ").strip() if self.text else ""
        return f"{self.type_name},{self.name},{self.time1},{self.time2},{self.price1},{self.price2},{self.color},{self.style},{escaped_text}"
