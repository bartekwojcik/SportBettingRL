from typing import List

from exp.training.params import ConfigParams
from exp.utils.plotting_utils import plot_results_error_bar
from exp.training.trainer import Trainer
from exp.training.trainer_logging_wrapper import LoggingTrainer
from exp.test_train_env_creator import TestTrainEnvCreator
from exp.evaluator import SimpleExperimentEvaluator
from exp.algorithm_registry import AlgorithmRegistry
from exp.utils.misc import validate_environments

from exp.utils.path_utils import (
    create_time_folder,
    create_run_folder,
    save_json_to_file,
)
import pprint
import neptune
from datetime import datetime
from exp.betting_env_creator import get_env_function


def experiment_main(
    root_save_path: str,
    video_record_test: bool,
    validate_env_render: bool,
    total_timesteps: int,
    n_eval_episodes: int,
    verbose: bool,
    render_test: bool,
    algorithms: List[str],
    neptune_api_token: str,
    norm_obs: bool,
    norm_reward: bool,
    episode_max_steps: int,
    norm_actions: bool,
    frame_stack: bool,
    path_to_keras_model: str,
    log: bool = True,
    path_to_data: str = None,
):
    (
        env_creating_function,
        env_name,
        train_env_parameters_dict,
        eval_env_parameters_dict,
        test_env_parameters_dict,
    ) = get_env_function(path_to_keras_model, path_to_data)

    validate_environments(env_creating_function(), not validate_env_render)
    test_train_env_creator = TestTrainEnvCreator()
    evaluator = SimpleExperimentEvaluator()

    run_save_path = create_time_folder(root_save_path)  # add timestamp

    all_algorithms_hyperparameters_map = AlgorithmRegistry.get_hyperparamets_per_algorithm(
        orignal_env=env_creating_function(), chosen_algorithms=algorithms
    )

    pprint.pprint("all algoritms and hyperparameters to be tested:")
    pprint.pprint(all_algorithms_hyperparameters_map)

    whole_experiment_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if log:
        save_json_to_file(
            report_save_path=run_save_path,
            file_name="all_algorithms_hyperparameters_map.json",
            **all_algorithms_hyperparameters_map,
        )
        neptune.init(
            project_qualified_name=f"asdd/sandbox", api_token=neptune_api_token,
        )

    results_holder = []

    for algorithm_key, hyperparameters_list in list(
        all_algorithms_hyperparameters_map.items()
    ):

        for index, hyperparameters in enumerate(hyperparameters_list):

            experiment_save_path = create_run_folder(
                run_save_path, algorithm_key, index
            )
            params_to_log = {
                "total_timesteps": total_timesteps,
                "episode_max_steps": episode_max_steps,
                "n_eval_episodes": n_eval_episodes,
                "algorithm_key": algorithm_key,
                "index": index,
                "env": env_name,
                "experiment_save_path": experiment_save_path,
                "whole_experiment_start_time": whole_experiment_start_time,
                "norm_obs": norm_obs,
                "norm_reward": norm_reward,
                "norm_actions": norm_actions,
                "frame_stack": frame_stack,
                "train_env_parameters_dict": train_env_parameters_dict,
                "eval_env_parameters_dict": eval_env_parameters_dict,
                "test_env_parameters_dict": test_env_parameters_dict,
            }

            trainer = Trainer()
            if log:
                trainer = LoggingTrainer(
                    trainer=trainer,
                    params_to_log=params_to_log,
                    algorithm_hyperparamters=hyperparameters,
                    experiment_save_path=experiment_save_path,
                    run_name=f"{algorithm_key}-{index}",
                )

            training_params = ConfigParams(algorithm_key=algorithm_key,
                                           algorithm_hyperparameters=hyperparameters,
                                           env_creating_function=env_creating_function,
                                           environment_creator=test_train_env_creator,
                                           save_folder=experiment_save_path,
                                           video_record_test=video_record_test,
                                           total_timesteps=total_timesteps,
                                           n_eval_episodes=n_eval_episodes,
                                           evaluator=evaluator,
                                           verbose=verbose,
                                           render_test=render_test,
                                           norm_obs=norm_obs,
                                           norm_reward=norm_reward,
                                           episode_max_steps=episode_max_steps,
                                           norm_actions=norm_actions,
                                           frame_stack=frame_stack,
                                           train_env_parameters_dict=train_env_parameters_dict,
                                           eval_env_parameters_dict=eval_env_parameters_dict,
                                           test_env_parameters_dict=test_env_parameters_dict)

            training_configuarion = trainer.prepare(params=training_params)

            mean_reward, std_reward, training_time_minutes = trainer.train(
                training_params=training_configuarion
            )

            results_holder.append(
                {
                    "name": f"{algorithm_key}-{index}",
                    "mean": mean_reward,
                    "std": std_reward,
                }
            )

    plot_results_error_bar(results_holder, run_save_path, "error_bar.jpg")
