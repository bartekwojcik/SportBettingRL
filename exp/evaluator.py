from abc import ABC, abstractmethod
from typing import Union, Tuple, List

import gym
from stable_baselines.common import BaseRLModel
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecEnv


class BaseExperimentEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        model: BaseRLModel,
        test_env: VecEnv,
        n_eval_episodes: int,
        render_test: bool,
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
        return None


class SimpleExperimentEvaluator(BaseExperimentEvaluator):
    def evaluate(
        self,
        model: BaseRLModel,
        test_env: gym.Env,
        n_eval_episodes: int,
        render_test: bool,
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
        mean_reward, std_reward = evaluate_policy(
            model, test_env, n_eval_episodes=n_eval_episodes, render=render_test
        )

        return mean_reward, std_reward
