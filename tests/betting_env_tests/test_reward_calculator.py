import unittest
from betting_env.reward_calculator import DecimalRewardCalculator

class TestRewardCalculator(unittest.TestCase):

    def test_positive_reward_when_winning(self):

        rc = DecimalRewardCalculator()

        stake = 10
        odds = 1.9
        won = True

        reward = rc.calculate_reward(bet_amount=stake,
                                     reward_odds=odds,
                                     won=won
                                     )

        assert reward == 9

    def test_negative_reward_when_losing(self):

        rc = DecimalRewardCalculator()

        stake = 10
        odds = 1.9
        won = False

        reward = rc.calculate_reward(bet_amount=stake,
                                     reward_odds=odds,
                                     won=won
                                     )

        assert reward == -10


    def test_reward_is_zero_when_odds_are_1_and_winning(self):

        rc = DecimalRewardCalculator()

        stake = 10
        odds = 1
        won = True

        reward = rc.calculate_reward(bet_amount=stake,
                                     reward_odds=odds,
                                     won=won
                                     )

        assert reward == 0

    def test_reward_is_zero_when_losing_and_betting_zero(self):

        rc = DecimalRewardCalculator()

        stake = 0
        odds = 1
        won = False

        reward = rc.calculate_reward(bet_amount=stake,
                                     reward_odds=odds,
                                     won=won
                                     )

        assert reward == 0

    def test_reward_is_zero_when_winning_and_betting_zero(self):

        rc = DecimalRewardCalculator()

        stake = 0
        odds = 1
        won = True

        reward = rc.calculate_reward(bet_amount=stake,
                                     reward_odds=odds,
                                     won=won
                                     )

        assert reward == 0

    def test_negative_result_when_smaller_than_0_odds(self):

        rc = DecimalRewardCalculator()

        stake = 10
        odds = 0.5
        won = True

        reward = rc.calculate_reward(bet_amount=stake,
                                     reward_odds=odds,
                                     won=won
                                     )

        assert reward == -5


