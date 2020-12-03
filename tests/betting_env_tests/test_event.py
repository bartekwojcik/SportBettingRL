import unittest
import os
from datetime import datetime
from betting_env.enums.bet_enum import WinnerEnum
from betting_env.event import Event


class TestEvent(unittest.TestCase):


    def test_event_indicates_home_won(self):
        event = Event(
            event_id=1,
            event_date=datetime(2018, 1, 2),
            away_team=13,
            home_team=56,
            league=9,
            away_score=1,
            home_score=2,
            away_odds=0.5,
            home_odds=1.3,
            draw_odds=5,
        )

        who_won = event.get_winner()

        assert who_won == WinnerEnum.home_won
        assert who_won != WinnerEnum.away_won
        assert who_won != WinnerEnum.draw_won



    def test_event_indicates_away_won(self):
        event = Event(
            event_id=1,
            event_date=datetime(2018, 1, 2),
            away_team=13,
            home_team=56,
            league=9,
            away_score=3,
            home_score=2,
            away_odds=0.5,
            home_odds=1.3,
            draw_odds=5,
        )

        who_won = event.get_winner()

        assert who_won != WinnerEnum.home_won
        assert who_won == WinnerEnum.away_won
        assert who_won != WinnerEnum.draw_won
    def test_event_indicates_draw_won(self):

        event = Event(
            event_id=1,
            event_date=datetime(2018, 1, 2),
            away_team=13,
            home_team=56,
            league=9,
            away_score=20,
            home_score=20,
            away_odds=0.5,
            home_odds=1.3,
            draw_odds=5,
        )

        who_won = event.get_winner()

        assert who_won != WinnerEnum.home_won
        assert who_won != WinnerEnum.away_won
        assert who_won == WinnerEnum.draw_won


