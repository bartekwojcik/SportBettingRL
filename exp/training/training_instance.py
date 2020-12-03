from stable_baselines.common.callbacks import CallbackList, EvalCallback
from typing import Union, Tuple, List
from stable_baselines.common.vec_env import VecEnv, VecNormalize

from exp.callbacks.save_vec_norm_callback import SaveVecNormalizeCallback
from exp.evaluator import BaseExperimentEvaluator
import os
from os.path import exists
import gym
from exp.callbacks.new_best_callback import NewBestCallback
import time
from typing import Tuple

from stable_baselines.common import BaseRLModel

from exp.training.params import TrainingParams


class TrainingInstance:

    def _get_best_or_last_model_path(
        self, last_vecnorm_path, best_vecnorm_path, is_normalized
    ):
        """
        Depending on whether there was any best model (if training was too short there will be none) pics right model
        paths to be loaded

        :return: path to model and to vecpath
        """
        best_model_path = os.path.join(self.BEST_MODEL_PATH, "best_model.zip")
        last_model_path = self.FINAL_MODEL_PATH

        if is_normalized:
            if exists(best_model_path) and exists(best_vecnorm_path):
                print("loading best model and normalized vecenc for testing")
                return best_model_path, best_vecnorm_path
            elif exists(last_model_path) and exists(last_vecnorm_path):
                print("loading last model and normalized vecenc for testing")
                return last_model_path, last_vecnorm_path
            else:
                raise ValueError("No model to load for testing.")
        else:
            if exists(best_model_path):
                print("loading best model for testing")
                return best_model_path, None
            elif exists(last_model_path):
                print("loading last model for testing")
                return last_model_path, None
            else:
                raise ValueError("No model to load for testing.")

    def _get_callbacks(
        self, eval_env, best_model_stats_path, best_model_path, eval_freq=40000
    ):
        save_vec_normalize = SaveVecNormalizeCallback(
            save_freq=1, save_path=best_model_stats_path
        )
        new_best_callback = NewBestCallback(log_dir=best_model_stats_path)
        eval_callback = EvalCallback(
            eval_env,
            log_path=best_model_stats_path,
            best_model_save_path=best_model_path,
            eval_freq=eval_freq,
            callback_on_new_best=CallbackList([save_vec_normalize, new_best_callback]),
        )

        callbacks = CallbackList([eval_callback])
        return callbacks, save_vec_normalize.get_env_path()

    def _set_paths(self, save_folder_path: str):
        self.FINAL_MODEL_PATH = os.path.join(save_folder_path, "last_model.zip")
        self.BEST_MODEL_PATH = os.path.join(save_folder_path, "best_model")
        self.BEST_MODEL_STATS_PATH = os.path.join(save_folder_path, "best_model_stats")

    def train(
        self,
        training_params:TrainingParams
    ) -> Union[Tuple[float, float, float], Tuple[List[float], List[int], float]]:
        """
        Starts training

        :param training_params:
        :return: mean_reward, std_reward, training_time_minutes
        """


        self._set_paths(training_params.results_save_folder)

        poor_mans_eval_freq = max(int((training_params.total_timesteps - 1) / 10000) - 1, 50)
        callbacks, best_vect_path = self._get_callbacks(
            eval_env=training_params.envs.eval_env,
            best_model_stats_path=self.BEST_MODEL_STATS_PATH,
            best_model_path=self.BEST_MODEL_PATH,
            eval_freq=poor_mans_eval_freq,
        )

        start = time.time()

        algorithm = training_params.algorithm

        algorithm.learn(total_timesteps=training_params.total_timesteps, callback=callbacks)
        end = time.time()

        train_env = training_params.envs.train_env
        test_env = training_params.envs.test_env

        algorithm.save(self.FINAL_MODEL_PATH)
        training_time_minutes = (end - start) / 60
        print(f"Training time: {training_params.algorithm_name} ", training_time_minutes, " minutes")

        save_vecnorm = (training_params.vecnorm_path is not None) and isinstance(train_env, VecNormalize)

        if save_vecnorm:
            train_env.save(training_params.vecnorm_path)  # saving best vectornormalisation anyway

        zip_file, chosen_vecnorm_path = self._get_best_or_last_model_path(
            last_vecnorm_path=training_params.vecnorm_path,
            best_vecnorm_path=best_vect_path,
            is_normalized=save_vecnorm,
        )
        model = algorithm.load(zip_file)

        if save_vecnorm:
            test_env = VecNormalize.load(chosen_vecnorm_path, test_env)
            test_env.training = False
            test_env.norm_reward = False

        mean_reward, std_reward = training_params.evaluator.evaluate(
            model, test_env, n_eval_episodes=training_params.n_eval_episodes, render_test=training_params.render_test
        )

        return mean_reward, std_reward, training_time_minutes
