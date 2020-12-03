from typing import Dict, Any

import neptune

from exp.training.params import ConfigParams
from exp.training.trainer import Trainer
from exp.utils.path_utils import save_json_to_file


class LoggingTrainer:
    def __init__(
        self,
        trainer: Trainer,
        algorithm_hyperparamters: Dict[str, Any],
        params_to_log: Dict[str, Any],
        experiment_save_path: str,
        run_name: str,
    ):
        self.experiment_save_path = experiment_save_path
        self.params_to_log = params_to_log
        self.algorithm_hyperparamters = algorithm_hyperparamters
        self.trainer = trainer
        self.run_name = run_name


    def prepare(self, **kwargs):

        full_paramters_as_dict = {}
        full_paramters_as_dict.update(self.params_to_log)
        full_paramters_as_dict.update(self.algorithm_hyperparamters)

        neptune.create_experiment(name=f"run_experiment", params=full_paramters_as_dict)

        save_json_to_file(
            report_save_path=self.experiment_save_path,
            file_name="params.json",
            **self.params_to_log,
        )
        save_json_to_file(
            report_save_path=self.experiment_save_path,
            file_name="algorithm_hyperparams.json",
            **self.algorithm_hyperparamters,
        )

        return self.trainer.prepare(**kwargs)

    def train(self, **kwargs):

        result = self.trainer.train(**kwargs)

        (
            mean_reward,
            std_reward,
            training_time_minutes,
        ) = result[0],result[1],result[2]

        metrics_to_save = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "training_time_minutes": training_time_minutes,
        }

        save_json_to_file(self.experiment_save_path, "metrics.json", **metrics_to_save)

        for k, v in metrics_to_save.items():
            neptune.log_metric(k, v)

        neptune.stop()

        return result
