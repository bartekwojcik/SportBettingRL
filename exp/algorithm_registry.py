from typing import Dict, Any, List

from stable_baselines.common.vec_env import VecEnv

from exp.utils.misc import product_dict
import gym
from gym.envs.classic_control import PendulumEnv
from gym.envs.toy_text import BlackjackEnv
from stable_baselines import (
    A2C,
    ACER,
    ACKTR,
    DDPG,
    DQN,
    HER,
    GAIL,
    PPO2,
    SAC,
    TD3,
    TRPO,
)

from stable_baselines.common import BaseRLModel


S_POLICY = "policy"
S_GAMMA = "gamma"
S_LR_SCHEDULE = "lr_schedule"


POLICIES = ["MlpPolicy"]
GAMMAS = [0.99]
LR_SCHEDULE = [
    "linear",
    "constant",
    "double_linear_con",
    "middle_drop",
    "double_middle_drop",
]


def A2C_hyperparameters():
    return {S_POLICY: POLICIES, S_LR_SCHEDULE: ["linear"]}


def ACER_hyperparameters():

    return {S_POLICY: POLICIES, S_LR_SCHEDULE: ["linear"]}


def ACKTR_hyperparameters():

    return {
        S_POLICY: POLICIES,
        S_LR_SCHEDULE: ["linear"]
        # S_GAMMA: GAMMAS,
    }


def DDPG_hyperparameters():
    return {
        S_POLICY: POLICIES,
        # S_GAMMA: GAMMAS,
    }


def HER_hyperparameters():
    return {
        S_POLICY: POLICIES,
        # S_GAMMA: GAMMAS,
    }


def DQN_hyperparameters():

    return {
        S_POLICY: POLICIES,
        # S_GAMMA: GAMMAS,
        # S_LR_SCHEDULE: ['linear']
    }


def GAIL_hyperparameters():
    return {
        S_POLICY: POLICIES,
        # S_GAMMA: GAMMAS,
    }


def PPO2_hyperparameters():

    return {
        S_POLICY: POLICIES,
        # S_GAMMA: GAMMAS
    }


def TD3_hyperparameters():
    return {
        S_POLICY: POLICIES,
        # S_GAMMA: GAMMAS,
    }


def SAC_hyperparameters():
    return {
        S_POLICY: POLICIES,
        # S_GAMMA: GAMMAS,
    }


def TRPO_hyperparameters():
    return {
        S_POLICY: POLICIES,
        # S_GAMMA: GAMMAS,
    }


def validate_parameters(algo, param_dicts, original_env):
    algo_name = algo.__name__
    for params in param_dicts:
        try:
            is_working = algo(env=original_env, **params)
        except Exception as ex:
            print(f"Error for algorithm: {algo_name}.", ex.__traceback__)
            raise
        assert is_working, f"can't instantiate algorithm: {algo_name} "

    return True


def validate_algorithm_names(chosen_keys, existing_keys):
    # throws exception if there is a name of algorithm that is not registered
    for chosen in chosen_keys:
        if not (chosen in existing_keys):
            raise ValueError(f"{chosen} algorithm is not registered")

    return True


class AlgorithmRegistry:
    "_algorithm_map-> KEY, (FUNCTION, HYPERPARAMERS DICT)"
    _algorithm_map = {
        "A2C": (A2C, A2C_hyperparameters),
        "ACER": (ACER, ACER_hyperparameters),
        "ACKTR": (ACKTR, ACKTR_hyperparameters),
        "DDPG": (DDPG, DDPG_hyperparameters),
        "DQN": (DQN, DQN_hyperparameters),
        # "HER": (HER,HER_hyperparameters), https://stable-baselines.readthedocs.io/en/master/modules/her.html
        # "GAIL": (GAIL, GAIL_hyperparameters), expert dataset is required
        "PPO": (PPO2, PPO2_hyperparameters),
        "SAC": (SAC, SAC_hyperparameters),
        "TD3": (TD3, TD3_hyperparameters),
        "TRPO": (TRPO, TRPO_hyperparameters),
    }

    @staticmethod
    def get_algorithms_map():
        return AlgorithmRegistry._algorithm_map

    @staticmethod
    def get_hyperparamets_per_algorithm(
        orignal_env: gym.Env, chosen_algorithms: List[str], validate=True
    ) -> Dict[str, Any]:
        """
        :param orignal_env:
        :param chosen_algorithms:
        :param validate:
        :return: Dictionary of algorithm_name->List[Dict[hyperparameters]].
        Example:
        {'A2C':
        [{'gamma': 0.99, 'momentum': 0.0, 'policy': 'MlpPolicy'},
             {'gamma': 0.999, 'momentum': 0.1, 'policy': 'MlpPolicy'}]}


        """
        result = {}

        if validate:

            validate_algorithm_names(
                chosen_algorithms, AlgorithmRegistry.get_algorithms_map().keys()
            )
            print("algorithms validated")

        for name, (algo, param_fun) in AlgorithmRegistry.get_algorithms_map().items():
            if not (name in chosen_algorithms):
                continue

            params_definition = param_fun()
            param_dicts = list(product_dict(**params_definition))
            result[name] = param_dicts
            if validate:
                validate_parameters(algo, param_dicts, orignal_env)

        return result

    @staticmethod
    def get_algorithm(
        algorithm_key: str,
        env: VecEnv,
        tensorboard_path: str,
        verbose: int,
        algorithm_hyperparameters: Dict[str, Any],
    ) -> BaseRLModel:
        algorithm_class = AlgorithmRegistry.get_algorithms_map()[algorithm_key][0]

        algorithm = algorithm_class(
            env=env,
            tensorboard_log=tensorboard_path,
            verbose=verbose,
            **algorithm_hyperparameters,
        )

        return algorithm
