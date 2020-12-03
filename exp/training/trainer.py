from typing import Dict, Any, Callable

from exp.evaluator import BaseExperimentEvaluator
from exp.test_train_env_creator import TestTrainEnvCreator
from exp.training.params import ConfigParams, TrainingParams
from exp.training.training_instance import TrainingInstance
from exp.algorithm_registry import AlgorithmRegistry
import os


class Trainer:
    def prepare(self, params: ConfigParams):
        """
        Prepares parameters required for training

        :param params:
        :return:
        """
        print(f"Algorithm: {params.algorithm_key}")
        print(f"Hyperparamters: {params.algorithm_hyperparameters}")

        video_record_path = params.save_folder if params.video_record_test else None

        if params.algorithm_key in ["DDPG", "DQN", "TD3"]:
            print(
                "Setting number of train environments to 1 because apparently",
                params.algorithm_key,
                "needs it",
            )
            params.n_train_env = 1

        (
            train_env,
            test_env,
            eval_env,
        ) = params.environment_creator.get_baselines_environment(
            env_creating_function=params.env_creating_function,
            video_recording_path=video_record_path,
            episode_max_steps=params.episode_max_steps,
            norm_obs=params.norm_obs,
            norm_reward=params.norm_reward,
            frame_stack=params.frame_stack,
            norm_actions=params.norm_actions,
            train_env_parameters_dict=params.env_parameters.train_env_parameters_dict,
            eval_env_parameters_dict=params.env_parameters.eval_env_parameters_dict,
            test_env_parameters_dict=params.env_parameters.test_env_parameters_dict,
            n_train_env=params.n_train_env,
        )

        tensorboard_path = os.path.join(params.save_folder, "tensorboard")

        algorithm = AlgorithmRegistry.get_algorithm(
            algorithm_key=params.algorithm_key,
            env=train_env,
            tensorboard_path=tensorboard_path,
            verbose=params.verbose,
            algorithm_hyperparameters=params.algorithm_hyperparameters,
        )
        algorithm_name = algorithm.__class__.__name__

        vecnorm_path = (
            os.path.join(params.save_folder, "last_vec_normalize.pkl")
            if (params.norm_obs or params.norm_reward)
            else None
        )

        return TrainingParams(
            algorithm=algorithm,
            algorithm_name=algorithm_name,
            results_save_folder=params.save_folder,
            total_timesteps=params.total_timesteps,
            train_env=train_env,
            test_env=test_env,
            eval_env=eval_env,
            evaluator=params.evaluator,
            render_test=params.render_test,
            n_eval_episodes=params.n_eval_episodes,
            vecnorm_path=vecnorm_path,
        )

    def train(self, **kwargs):
        """
        Starts training procedure

        :param kwargs:
        :return:
        """

        trainer = TrainingInstance()
        mean_reward, std_reward, training_time_minutes = trainer.train(
            **kwargs
        )

        print("mean_reward:", mean_reward)
        print("std_reward:", std_reward)

        return mean_reward, std_reward, training_time_minutes
