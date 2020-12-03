from typing import Dict, Any, Callable

from stable_baselines.common import BaseRLModel
from stable_baselines.common.vec_env import VecEnv

from exp.evaluator import BaseExperimentEvaluator
from exp.test_train_env_creator import TestTrainEnvCreator


class EnvParameters:
    def __init__(
        self,
        train_env_parameters_dict: Dict[str, Any],
        eval_env_parameters_dict: Dict[str, Any],
        test_env_parameters_dict: Dict[str, Any],
    ):
        self.test_env_parameters_dict = test_env_parameters_dict
        self.eval_env_parameters_dict = eval_env_parameters_dict
        self.train_env_parameters_dict = train_env_parameters_dict


class ConfigParams:
    """
    Plain object for storing paramters - avoids passing 100 parameters into every method
    """

    def __init__(
        self,
        algorithm_key: str,
        algorithm_hyperparameters: Dict[str, Any],
        env_creating_function: Callable,
        environment_creator: TestTrainEnvCreator,
        save_folder: str,
        video_record_test: bool,
        total_timesteps: int,
        render_test: bool,
        n_eval_episodes: int,
        evaluator: BaseExperimentEvaluator,
        verbose: bool,
        norm_obs: bool,
        norm_reward: bool,
        norm_actions: bool,
        frame_stack: bool,
        episode_max_steps: int,
        train_env_parameters_dict: Dict[str, Any],
        eval_env_parameters_dict: Dict[str, Any],
        test_env_parameters_dict: Dict[str, Any],
        n_train_env=4,
    ):

        self.algorithm_key = algorithm_key
        self.algorithm_hyperparameters = algorithm_hyperparameters
        self.env_creating_function = env_creating_function
        self.environment_creator = environment_creator
        self.save_folder = save_folder
        self.video_record_test = video_record_test
        self.total_timesteps = total_timesteps
        self.render_test = render_test
        self.n_eval_episodes = n_eval_episodes
        self.evaluator = evaluator
        self.verbose = verbose
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.norm_actions = norm_actions
        self.frame_stack = frame_stack
        self.episode_max_steps = episode_max_steps
        self.n_train_env = n_train_env

        self.env_parameters = EnvParameters(
            train_env_parameters_dict,
            eval_env_parameters_dict,
            test_env_parameters_dict,
        )


class EnvHolder:
    def __init__(
        self, train_env: VecEnv, test_env: VecEnv, eval_env: VecEnv,
    ):
        self.eval_env = eval_env
        self.test_env = test_env
        self.train_env = train_env


class TrainingParams:
    def __init__(
        self,
        algorithm: BaseRLModel,
        algorithm_name: str,
        results_save_folder,
        total_timesteps: int,
        train_env: VecEnv,
        test_env: VecEnv,
        eval_env: VecEnv,
        evaluator: BaseExperimentEvaluator,
        render_test=True,
        n_eval_episodes=5,
        vecnorm_path: str = None,
    ):
        self.vecnorm_path = vecnorm_path
        self.n_eval_episodes = n_eval_episodes
        self.render_test = render_test
        self.evaluator = evaluator

        self.total_timesteps = total_timesteps
        self.results_save_folder = results_save_folder
        self.algorithm_name = algorithm_name
        self.algorithm = algorithm

        self.envs = EnvHolder(train_env, test_env, eval_env)
