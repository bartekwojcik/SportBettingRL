import gym
import numpy as np
import typing


class KerasWrapperEnv(gym.Env):
    """
    Wrapper for Betting Environment that uses pretrained keras model to predict which team will winn based only on bookmakers odds.
    """

    def __init__(self, env: gym.Env, keras_model):
        """
        :param env: original env
        :param keras_model: loaded keras model
        """
        self.env = env
        self.keras_model = keras_model
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(
        self, action_int: int
    ) -> typing.Tuple[np.ndarray, float, bool, typing.Dict]:
        state, total_reward, done, _ = self.env.step(action_int)
        norm_odds = state[1:]
        normalized_bankroll = state[0].reshape(-1,)
        prediction = self.keras_model.predict(
            np.array(norm_odds).reshape(1, -1)
        ).reshape(-1,)

        vector_with_predictions = np.concatenate((normalized_bankroll, prediction))

        return vector_with_predictions, total_reward, done, _

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode)
