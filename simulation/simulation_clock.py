from datetime import datetime, timezone, timedelta

class SimulationClock:
    def __init__(self, start_time: datetime = None):
        self._current_time = start_time or datetime(2024, 1, 1, tzinfo=timezone.utc)

    def current_time(self) -> datetime:
        return self._current_time

    def set_time(self, time: datetime):
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)
        self._current_time = time

    def advance(self, delta: timedelta):
        self._current_time += delta

    def reset(self, start_time: datetime):
        self.set_time(start_time)
