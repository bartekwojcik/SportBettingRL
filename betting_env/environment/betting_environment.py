import numpy as np
import gym
import typing
from betting_env.event_factory import EventFactory
import gym.spaces as spaces
from betting_env.enums.bet_enum import WinnerEnum, ActionEnum
from betting_env.reward_calculator import DecimalRewardCalculator
from betting_env.utils import convert_event_to_state
from betting_env.state import EnvState
from betting_env.environment.actions import get_num_actions, int_to_actions
import random
import math


class BettingEnv(gym.Env):
    """Betting environment"""

    def __init__(
        self,
        event_factory: EventFactory,
        bankroll: float,
        reward_calculator: DecimalRewardCalculator,
        losing_limit: float = 0.0,
        winning_limit:float = 200,
        seed: int = 0,
        seed_end_range:int = 20000

    ):
        self.seed_end_range = seed_end_range
        self.winning_limit = winning_limit
        self.seed = seed
        self.reward_calculator = reward_calculator
        self.LOSING_LIMIT = losing_limit
        self.INITIAL_BANKROLL = bankroll
        self.bankroll = bankroll
        self.event_factory = event_factory
        self.current_event_index = seed
        self.state = self._get_state_by_index(self.current_event_index)

        n_actions = get_num_actions()
        self.action_space = spaces.Discrete(n_actions)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self._game_ended = False

    def _does_epoch_end(self) -> bool:
        """
        Check if any condition for wining or losing has been met
        :return: Bool
        """
        agent_lost_money = self.bankroll <= 0.0 or self.bankroll <= self.LOSING_LIMIT

        is_next_event_available = self.event_factory.is_event_available(
            self.current_event_index + 1
        )
        agent_reached_wining_limit = self.bankroll >= self.winning_limit
        if agent_reached_wining_limit:
            print('wooooooooow, agent reached winning limit. Bankroll:', self.bankroll, 'Limit: ', self.winning_limit)
        return agent_lost_money or (not is_next_event_available) or agent_reached_wining_limit

    def _did_agent_win_bet(
        self,
        action: ActionEnum,
        winner: WinnerEnum,
        home_odds: float,
        away_odds: float,
        draw_odds: float,
    ) -> typing.Tuple[bool, float]:
        """
        Checks if agent win given bet
        :param action: Action undertaken by a player
        :param winner: Winner of the match (home, away, draw)
        :param home_odds: Odds of home team winning
        :param away_odds: Odds of away team winning
        :param draw_odds: Odds of draw
        :return: Boolean whethear or not agent won and float of money reward
        """
        won = False
        reward_odds = 0.0

        if winner == WinnerEnum.home_won and action == action.bet_home:
            won = True
            reward_odds = home_odds

        elif winner == WinnerEnum.away_won and action == action.bet_away:
            won = True
            reward_odds = away_odds

        elif winner == WinnerEnum.draw_won and action == action.bet_draw:
            won = True
            reward_odds = draw_odds

        return won, reward_odds

    def _calculate_reward(
        self,
        bet_amount: float,
        action: ActionEnum,
        who_won_event: WinnerEnum,
        home_odds: float,
        away_odds: float,
        draw_odds: float,
    ) -> float:
        """
        Calculate a reward for agent's action
        :param bet_amount: Amount of money betted by agent
        :param action: Agents action
        :param who_won_event: A real winner of a match
        :param home_odds: Odds of home team winning
        :param away_odds: Odds of away team winning
        :param draw_odds: Odds of draw
        :return: float of money reward
        """

        (won, reward_odds) = self._did_agent_win_bet(
            action, who_won_event, home_odds, away_odds, draw_odds
        )

        total_reward = self.reward_calculator.calculate_reward(
            bet_amount, reward_odds, won
        )

        return total_reward

    def step(
        self, action_int: int
    ) -> typing.Tuple[EnvState, float, bool, typing.Dict]:
        """
        Step

        :param action_int: index of an action according to :func:`~betting_env/environment/actions.py/action_map`
        :return: state, reward, done, config
        """

        if self._game_ended:
            raise ValueError("Game as already ended")
        action_tuple = int_to_actions(action_int)
        action = action_tuple[0]
        bet_amount = action_tuple[1][0]
        bet_amount = bet_amount if bet_amount >= 0 else 0
        bet_amount = bet_amount if bet_amount <= self.bankroll else self.bankroll

        action = ActionEnum(action)
        total_reward = 0

        if action == ActionEnum.do_nothing:
            pass  # do nothing, do not update bankroll
        else:
            home_odds = self.state.home_odds
            away_odds = self.state.away_odds
            draw_odds = self.state.draw_odds
            event_winner = self.state.get_original_event().get_winner()

            total_reward = self._calculate_reward(
                bet_amount, action, event_winner, home_odds, away_odds, draw_odds
            )

            # update Bankroll
            self.bankroll += total_reward

        done = self._does_epoch_end()

        if not done:
            self._set_new_state()
        if done:
            self._game_ended = True

        # todo replace to_small_vector with normal
        return self.state, total_reward, done, {}

    def _get_state_by_index(self, index: int) -> EnvState:
        """
        Gets state from event factory

        :param index: index of event
        :return: EnvState
        """

        event = self.event_factory.get_event_by_index(index)
        current_state = convert_event_to_state(event, self.bankroll)
        return current_state

    def _set_new_state(self) -> None:
        """
        Sets next state (increments index by one)

        :return: None
        """
        self.current_event_index += 1
        self.state = self._get_state_by_index(self.current_event_index)

    def reset(self) -> EnvState:
        """
        Resets environment

        :return: first state
        """
        self._game_ended = False
        self.bankroll = self.INITIAL_BANKROLL

        end = self.seed + self.seed_end_range
        new_start = random.randint(self.seed, end)
        self.current_event_index = new_start#self.seed
        self.state = self._get_state_by_index(self.current_event_index)

        # todo replace to_small_vector with normal
        return self.state

    def render(self, mode="human"):
        raise NotImplementedError()
        # perhaps return bankroll?
