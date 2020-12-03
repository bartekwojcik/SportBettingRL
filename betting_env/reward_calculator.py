import math


def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n


class DecimalRewardCalculator:
    def calculate_reward(
        self, bet_amount: float, reward_odds: float, won: bool
    ) -> float:
        """
        it returns a total reward, namely if you bet 50 for losing team your 'reward' is not zero but -50 (your stake does not return to you).
        however if you bet 50 and won 59 (50 * odds) your reward is 9, a difference of what you had before a bet and what you have after learning a result of a bet.
        :param bet_amount: amount of money
        :param reward_odds: winning odds
        :param won: whether or not a bet was successful
        :return: reward
        """

        won = int(won)
        investment = bet_amount
        payout = (bet_amount * reward_odds) * won

        total_reward = payout - investment

        truncated = truncate(total_reward, 2)

        return truncated


class NormalizedRewardCalculator:
    def calculate_reward(
            self, bet_amount: float, reward_odds: float, won: bool
    ) -> float:
        #problem here is that this calculator does not return real money to the bankroll -
        #it was supposed to reuturn real reward to environment to update bankroll and normalised to Algorithm but forgot about the first part 11
        won = int(won)
        investment = bet_amount
        payout = (bet_amount * reward_odds) * won

        total_reward = payout - investment

        truncated = truncate(total_reward, 2)

        normalized = (truncated - bet_amount) / bet_amount if bet_amount != 0 else truncated

        return normalized
