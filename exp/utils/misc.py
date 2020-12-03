import itertools
import argparse

import gym
from stable_baselines.common.env_checker import check_env


def product_dict(**kwargs):
    """
    #https://stackoverflow.com/a/5228294/2710943
    usage: list(product_dict(**mydict)
    :param kwargs:
    :return:
    """

    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def str2bool(v):
    """
    Converts string to boolean for argparse
    :param v:
    :return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def validate_environments(env: gym.Env, skip_render_check):

    check_env(env, warn=True, skip_render_check=skip_render_check)
