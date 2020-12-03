import os
import unittest
from betting_env.environment.actions import action_map, action_to_int
from betting_env.data_loading_strategies.csv_loading_strategy import CSVLoadingStrategy
from betting_env.data_loading_strategies.csv_column_mapping import CsvColumnsMappings
from betting_env.data_preprocessor import DataPreprocessor
from betting_env.event_factory import EventFactory
from betting_env.environment.betting_environment import BettingEnv
from betting_env.reward_calculator import DecimalRewardCalculator
from betting_env.enums.bet_enum import ActionEnum, WinnerEnum
import numpy as np


class TestBettingEnvironment(unittest.TestCase):

    def _get_environment(self, bankroll=50, losing_limit=0, max_bet=100):
        env = BettingEnv(event_factory=self.event_factory,
                         bankroll=bankroll,
                         reward_calculator=self.reward_calculator,
                         losing_limit=losing_limit,
                         seed=0,
                         seed_end_range=0
                         )
        return env

    @classmethod
    def setUpClass(cls) -> None:
        PATH_TO_DATA = os.path.join(".", "resources", "kaggle_example_betting_csv.txt")
        col_map = CsvColumnsMappings(
            event_id='match_id',
            event_date='match_date',
            away_team='away_team',
            home_team='home_team',
            league='league',
            away_score='away_score',
            home_score='home_score',
            away_odds='avg_odds_away_win',
            home_odds='avg_odds_home_win',
            draw_odds='avg_odds_draw'
        )
        loading_strategy = CSVLoadingStrategy(PATH_TO_DATA, col_map)
        df = loading_strategy.dataframe

        columns_to_encode = ['league', 'home_team', 'away_team']
        preprocessor = DataPreprocessor()
        encoded_df, col_to_encoder_map = preprocessor.encode_columns(df, columns_to_encode)
        cls.col_to_encoder_map = col_to_encoder_map
        cls.event_factory = EventFactory(encoded_df.copy(), col_map)
        cls.reward_calculator = DecimalRewardCalculator()
        _ = cls.event_factory.get_all_events()

    def setUp(self) -> None:
        self.env = self._get_environment()
        self.env.reset()

    def test_epoch_using_kaggle_data_should_be_in_order(self):

        state = self.env.reset()

        assert state.home_team_id == 3 #
        assert state.away_team_id == 3 #
        assert state.event_day == 2 #.event_day
        assert state.event_month == 1
        assert state.event_year == 2005
        assert state.get_original_event().get_winner() == WinnerEnum.home_won

        action = action_map[0]
        action_index = action_to_int(action)
        state, reward, done, _ = self.env.step(action_index)

        assert done is False
        assert state.home_team_id == 0
        assert state.away_team_id == 1
        assert state.event_day == 1
        assert state.event_month == 2
        assert state.event_year == 2005
        assert state.get_original_event().get_winner() == WinnerEnum.home_won

        state, reward, done, _ = self.env.step(action_index)

        assert done is False
        assert state.home_team_id == 1
        assert state.away_team_id == 4
        assert state.event_day == 1
        assert state.event_month == 2
        assert state.event_year == 2005
        assert state.get_original_event().get_winner() == WinnerEnum.draw_won

        state, reward, done, _ = self.env.step(action_index)

        assert done is False
        assert state.home_team_id == 2
        assert state.away_team_id == 0
        assert state.event_day == 3
        assert state.event_month == 3
        assert state.event_year == 2005
        assert state.get_original_event().get_winner() == WinnerEnum.away_won

        state, reward, done, _ = self.env.step(action_index)

        assert done is False
        assert state.home_team_id == 4
        assert state.away_team_id == 2
        assert state.event_day == 1
        assert state.event_month == 1
        assert state.event_year == 2006
        assert state.get_original_event().get_winner() == WinnerEnum.away_won

        state, reward, done, _ = self.env.step(action_index)
        assert done is True

        with self.assertRaises(ValueError) as cm:
            self.env.step(action)

    def test_rewards_are_calculated_properly(self):

        previous_bankroll = self.env.bankroll
        action = action_map[5]
        expected_reward = 0.09
        action_index = action_to_int(action)
        state, reward, done, _ = self.env.step(action_index)

        assert reward == expected_reward
        assert self.env.bankroll == previous_bankroll + expected_reward

        previous_bankroll = self.env.bankroll
        action = action_map[0]
        expected_reward = (-1) * (action[1][0])
        action_index = action_to_int(action)
        state, reward, done, _ = self.env.step(action_index)

        assert reward == expected_reward
        assert self.env.bankroll == previous_bankroll + expected_reward

        previous_bankroll = self.env.bankroll
        action = action_map[10]
        expected_reward = (-1) * (action[1][0])
        action_index = action_to_int(action)
        state, reward, done, _ = self.env.step(action_index)

        assert reward == expected_reward
        assert self.env.bankroll == previous_bankroll + expected_reward


    def test_max_bet_cant_be_bigger_than_current_bankroll(self):
        env = self._get_environment(bankroll=1)

        action = action_map[5]
        expected_reward = 0.09
        action_index = action_to_int(action)
        state, reward, done, _ = env.step(action_index)

        assert expected_reward == reward

    def test_reset_is_correct(self):
        initial_state = self.env.state
        initial_bankroll = self.env.bankroll

        assert self.env._game_ended is False  # reference to private attribute?
        assert initial_state.home_team_id == 3
        assert initial_state.away_team_id == 3
        assert initial_state.event_day == 2
        assert initial_state.event_month == 1
        assert initial_state.event_year == 2005
        assert initial_state.get_original_event().get_winner() == WinnerEnum.home_won

        action = action_map[5]
        action_index = action_to_int(action)
        new_state, reward, done, _ = self.env.step(action_index)

        assert not (new_state.to_vector() != initial_state.to_vector()).all()
        assert initial_bankroll != self.env.bankroll

        self.env.reset()

        reset_state = self.env.state
        reset_bankroll = self.env.bankroll

        assert (reset_state.to_vector() == initial_state.to_vector()).all()
        assert not (reset_state.to_vector() != new_state.to_vector()).all()

        assert reset_bankroll == initial_bankroll


    def test_bankroll_is_increasing_by_proper_ammount_when_winning(self):

        previous_bankroll = self.env.bankroll
        # state = self.env.state
        # real_winner = WinnerEnum.home_won
        # winning_odds =1.9456
        previous_bankroll = self.env.bankroll
        action = action_map[0]
        expected_reward = (-1) * (action[1][0])
        action_index = action_to_int(action)
        state, reward, done, _ = self.env.step(action_index)

        assert reward == expected_reward
        assert self.env.bankroll == previous_bankroll + expected_reward
        assert self.env.bankroll < previous_bankroll

        previous_bankroll = self.env.bankroll
        action = action_map[5]
        expected_reward = 0.08
        action_index = action_to_int(action)
        state, reward, done, _ = self.env.step(action_index)

        assert reward == expected_reward
        assert self.env.bankroll == previous_bankroll + expected_reward
        assert self.env.bankroll > previous_bankroll

    def test_bankroll_is_decreasing_by_proper_amount_when_losing(self):

        previous_bankroll = self.env.bankroll
        action = action_map[0]
        expected_reward = (-1) * (action[1][0])
        action_index = action_to_int(action)
        state, reward, done, _ = self.env.step(action_index)

        assert reward == expected_reward
        assert self.env.bankroll == previous_bankroll + expected_reward
        assert self.env.bankroll < previous_bankroll

        previous_bankroll = self.env.bankroll
        action = action_map[0]
        expected_reward = (-1) * (action[1][0])
        action_index = action_to_int(action)
        state, reward, done, _ = self.env.step(action_index)

        assert reward == expected_reward
        assert self.env.bankroll == previous_bankroll + expected_reward
        assert self.env.bankroll < previous_bankroll

    def test_epoch_ends_when_bankroll_zero(self):

        env = self._get_environment(bankroll=5)

        action = action_map[14]
        expected_reward = (-1) * (action[1][0])
        action_index = action_to_int(action)
        state, reward, done, _ = env.step(action_index)

        assert reward == expected_reward
        assert done is True

    def test_epoch_ends_when_losing_limit_reached(self):

        env = self._get_environment(bankroll=55, losing_limit=50)

        action = action_map[14]
        expected_reward = (-1) * (action[1][0])
        action_index = action_to_int(action)
        state, reward, done, _ = env.step(action_index)

        assert reward == expected_reward
        assert done is True

    def test_epoch_ends_when_no_more_events(self):

        n_events = len(self.event_factory.get_all_events())
        for i in range(n_events):
            action = action_map[14]
            expected_reward = (-1) * (action[1][0])
            action_index = action_to_int(action)
            state, reward, done, _ = self.env.step(action_index)

            if i == 4:
                assert done == True
            else:
                assert done == False



