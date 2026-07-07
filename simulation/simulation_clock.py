from datetime import datetime, timezone, timedelta

class SimulationClock:
    """
    Purpose:
        Manages the virtual timeline for the simulation. Ensures that
        all time-dependent framework logic uses historical timestamps.
    """
    def __init__(self, start_time: datetime):
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        self._current_time = start_time

    def current_time(self) -> datetime:
        """
        Purpose:
            Returns the current virtual time.

        Returns:
            datetime: Current UTC timestamp in the simulation.
        """
        return self._current_time

    def advance(self, delta: timedelta):
        self._current_time += delta

    def set_time(self, new_time: datetime):
        """
        Purpose:
            Updates the virtual clock to a specific timestamp.

        Arguments:
            new_time (datetime): The new virtual time.
        """
        if new_time.tzinfo is None:
            new_time = new_time.replace(tzinfo=timezone.utc)
        self._current_time = new_time

    def reset(self, start_time: datetime):
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        self._current_time = start_time
