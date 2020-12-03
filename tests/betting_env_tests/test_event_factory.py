import pandas as pd
import unittest
import os
from datetime import datetime
from betting_env.event_factory import EventFactory
from betting_env.data_loading_strategies.csv_column_mapping import CsvColumnsMappings
from betting_env.event import Event


class TestEventFactory(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        PATH_TO_DATA = os.path.join(".", "resources", "kaggle_example_betting_csv.txt")
        cls.col_map = CsvColumnsMappings(
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
        cls.df = pd.read_csv(PATH_TO_DATA, sep='\t')
        df_copy = cls.df.copy()
        # todo perhaps i want it in setUpClass?
        cls.event_factory = EventFactory(df_copy, cls.col_map)

    def test_events_are_in_same_order_as_df(self):

        events = self.event_factory.get_all_events()
        df = self.df

        assert len(events) > 0
        assert len(events) == len(df)

        for i, event in enumerate(events):
            df_row = df.iloc[i]
            df_index = df_row[self.col_map.event_id]

            assert df_index == event.event_id



    def test_csv_to_data_mapping_is_correct(self):

        first_row = self.event_factory.get_all_events()[0]
        fr = first_row
        assert str(fr.event_id) == '170088'
        assert fr.league == 'England: Premier League'
        assert fr.event_date == '01/01/2006' #t
        assert fr.home_team == 'Liverpool'
        assert fr.away_team == 'Chelsea'
        assert fr.away_score == 1
        assert fr.home_score == 0
        assert fr.home_odds == 2.9944
        assert fr.draw_odds == 3.1944
        assert fr.away_odds == 2.2256

    def test_is_next_event_is_available(self):

        all_events =  self.event_factory.get_all_events()
        expected_number_of_events = 5

        num_events = len(all_events)

        assert num_events == expected_number_of_events

        for i in range(num_events):
            event = self.event_factory.get_event_by_index(i)
            assert event is not None and isinstance(event, Event)

    def test_next_event_not_available(self):

        for i in range(6,10):
            with self.assertRaises(ValueError) as cm:
                self.event_factory.get_event_by_index(i)








