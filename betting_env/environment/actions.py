import numpy as np
from betting_env.enums.bet_enum import ActionEnum
from typing import Tuple

action_map = {
    0: (ActionEnum.bet_draw, np.array([0.1])),
    1: (ActionEnum.bet_draw, np.array([0.5])),
    2: (ActionEnum.bet_draw, np.array([1])),
    3: (ActionEnum.bet_draw, np.array([2])),
    4: (ActionEnum.bet_draw, np.array([5])),
    5: (ActionEnum.bet_home, np.array([0.1])),
    6: (ActionEnum.bet_home, np.array([0.5])),
    7: (ActionEnum.bet_home, np.array([1])),
    8: (ActionEnum.bet_home, np.array([2])),
    9: (ActionEnum.bet_home, np.array([5])),
    10: (ActionEnum.bet_away, np.array([0.1])),
    11: (ActionEnum.bet_away, np.array([0.5])),
    12: (ActionEnum.bet_away, np.array([1])),
    13: (ActionEnum.bet_away, np.array([2])),
    14: (ActionEnum.bet_away, np.array([5])),
}


def get_num_actions() -> int:
    return len(action_map)


def int_to_actions(value: int) -> Tuple[int, float]:
    return action_map[value]


def action_to_int(action_tuple) -> int:
    action_index = next(
        (
            index
            for index, dict_action in action_map.items()
            if action_tuple == dict_action
        ),
        None,
    )
    return action_index
