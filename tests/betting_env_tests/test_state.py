import unittest
import os
from datetime import datetime
from betting_env.event import Event
from betting_env.state import EnvState


class TestEnvironmentState(unittest.TestCase):
    def setUp(self) -> None:
        self.event = Event(
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
        self.current_bankroll = 100

        self.state = EnvState(
            home_team_id=self.event.home_team,
            away_team_id=self.event.away_team,
            league_id=self.event.league,
            event_day=self.event.event_date.day,
            event_month=self.event.event_date.month,
            event_year=self.event.event_date.year,
            home_odds=self.event.home_odds,
            away_odds=self.event.away_odds,
            draw_odds=self.event.draw_odds,
            current_bankroll=self.current_bankroll,
            original_event=self.event,
        )

    def test_state_to_vector_transformation(self):

        vector = self.state.to_vector()

        assert vector[0] == self.current_bankroll
        assert vector[1] == self.event.draw_odds
        assert vector[2] == self.event.away_odds
        assert vector[3]== self.event.home_odds
        assert vector[4]== self.event.event_date.year
        assert vector[5]== self.event.event_date.month
        assert vector[6]== self.event.event_date.day
        assert vector[7]== self.event.league
        assert vector[8]== self.event.away_team
        assert vector[9]== self.event.home_team


    def test_state_reads_original_event_correctly(self):

        retrieved_event = self.state.get_original_event()

        assert retrieved_event == self.event
