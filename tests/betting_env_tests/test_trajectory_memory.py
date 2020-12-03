import unittest
from betting_env.trajectory_memory import TrajectoryMemory
import numpy as np


class TestTrajectoryMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = TrajectoryMemory()

        self.trajectories = [
            (2, 3,  False, ),
            (8, 9,  False, ),
            (15, 16,  False, ),
            (21, 22,  True, ),
            (27, 28,  False, ),
            (32, 33,  False, 36),
            (38, 39,  False, ),
            (43, 44,  False, ),
            (48, 49,  True, ),
            (53, 54,  False, ),
            (62, 63, False, ),
            (72, 73, False, ),
        ]

    def test_memory_saved_properly(self):

        for tr in self.trajectories:
            self.memory.record_action(tr[0], tr[1], tr[2])

        # because last transition is not done (done==False) ( 72, 73, 74, False, 76)
        self.memory.set_done()

        # should be 3 trajectories
        assert len(self.memory.get_memory()) == 3

        # flattening memories so we can easily iterate over them and compare to list above
        concatenated_saved_memory = np.concatenate(self.memory.get_memory()).reshape(-1, 4)

        for i in range(len(self.trajectories) - 1):
            true_trajectory = self.trajectories[i]
            saved_trajectory = concatenated_saved_memory[i]

            assert isinstance(saved_trajectory, np.ndarray)

            assert true_trajectory[0] == saved_trajectory[1]
            assert true_trajectory[1] == saved_trajectory[2]
            assert true_trajectory[2] == saved_trajectory[3]


