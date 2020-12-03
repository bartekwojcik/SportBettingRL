from typing import Any
import numpy as np


class TrajectoryMemory:
    """
    Memorizes event_index -> action -> rewards -> done tuples
    """
    def __init__(self):
        self._memory = []
        self._current_trajectory = []

    def get_memory(self):
        return self._memory

    def _create_record(
        self, step_index: int, action: Any, reward: float, done: bool,
    ):
        record = [step_index, action, reward, done]
        return record

    def _get_current_step(self):
        return len(self._current_trajectory)

    def record_action(
        self, action: Any, reward: float, done: bool,
    ):
        current_step = self._get_current_step()

        record = self._create_record(current_step, action, reward, done)
        self._current_trajectory.append(record)

        if done:
            self._set_new_trajectory()

    def set_done(self):
        """In case for some reason you want to end current trajectory and start new one"""
        self._set_new_trajectory()

    def _set_new_trajectory(self):

        if len(self._current_trajectory) > 0:
            trajectory = np.array(self._current_trajectory, dtype=object)
            self._memory.append(trajectory)

        self._current_trajectory = []
