import gym
import numpy as np
import typing
from betting_env.environment.betting_environment import BettingEnv

class NormalizeStateWrapper(gym.Env):
    """
    Wrapper for Betting Environment that uses pretrained keras model to predict which team will winn based only on bookmakers odds.
    """

    def __init__(
            self,
            env: BettingEnv
    ):
        """
        :param env: original env
        :param keras_model: loaded keras model
        """
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(
            self, action_int: int
    ) -> typing.Tuple[np.ndarray, float, bool, typing.Dict]:
        state, total_reward, done, _ = self.env.step(action_int)

        return state.to_normalized_vector(self.env.INITIAL_BANKROLL), total_reward, done, _

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        return state.to_normalized_vector(self.env.INITIAL_BANKROLL)

    def render(self, mode="human"):
        return self.env.render(mode)
