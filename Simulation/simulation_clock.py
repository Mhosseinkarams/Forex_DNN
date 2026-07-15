from datetime import datetime, timezone, timedelta

class SimulationClock:
    def __init__(self, start_time: datetime):
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        self._current_time = start_time

    def current_time(self) -> datetime:
        return self._current_time

    def advance(self, delta: timedelta):
        self._current_time += delta

    def set_time(self, new_time: datetime):
        if new_time.tzinfo is None:
            new_time = new_time.replace(tzinfo=timezone.utc)
        self._current_time = new_time

    def reset(self, start_time: datetime):
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        self._current_time = start_time
