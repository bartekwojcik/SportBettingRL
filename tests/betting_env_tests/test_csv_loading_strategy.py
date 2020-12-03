import unittest
import pandas as pd
import os
from datetime import datetime
from betting_env.data_loading_strategies.csv_loading_strategy import CSVLoadingStrategy
from betting_env.data_loading_strategies.csv_column_mapping import CsvColumnsMappings


class TestCsvLoadingStrategy(unittest.TestCase):
    def setUp(self) -> None:
        PATH_TO_DATA = os.path.join(".", "resources", "kaggle_example_betting_csv.txt")
        self.col_map = CsvColumnsMappings(
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
        self.loader = CSVLoadingStrategy(PATH_TO_DATA, self.col_map)

        #todo should test it using not Events but pandas DF
        #these tests should be in Event factory

    def test_data_is_in_chronological_order(self):
        """lines should be in chronological order."""

        dataframe = self.loader.dataframe

        assert len(dataframe) == 5, "no events"
        assert dataframe.iloc[0][self.col_map.event_date] == datetime(2005,1,2)
        assert dataframe.iloc[1][self.col_map.event_date] == datetime(2005,2,1)
        assert dataframe.iloc[2][self.col_map.event_date] == datetime(2005,2,1)
        assert dataframe.iloc[3][self.col_map.event_date] == datetime(2005,3,3)
        assert dataframe.iloc[4][self.col_map.event_date] == datetime(2006, 1, 1)

    def test_csv_to_data_mapping_is_correct(self):
        #todo
        #VERY LIKELY YOU WILL HAVE TO CHANGE TEAMS IDs to match data preprocessor
        first_row = self.loader.dataframe.iloc[0]
        fr = first_row
        assert str(fr[self.col_map.event_id]) == '170089'
        assert fr[self.col_map.league] == 'England: Premier League'
        assert fr[self.col_map.event_date] == datetime(2005,1,2)
        assert fr[self.col_map.home_team] == 'Fulham'
        assert fr[self.col_map.away_team] == 'Crystal Palace'
        assert fr[self.col_map.away_score] == 1
        assert fr[self.col_map.home_score] == 3
        assert fr[self.col_map.home_odds] == 1.9456
        assert fr[self.col_map.draw_odds] == 3.2333
        assert fr[self.col_map.away_odds] == 3.6722



    # def test_data_is_in_chronological_order(self):
    #     """lines should be in chronological order."""
    #
    #     events = self.loader.get_events()
    #
    #     assert len(events) == 5, "no events"
    #     assert events[0].event_date == datetime(2005,1,2)
    #     assert events[1].event_date == datetime(2005,2,1)
    #     assert events[2].event_date == datetime(2005,2,1)
    #     assert events[3].event_date == datetime(2005,3,3)
    #     assert events[4].event_date == datetime(2006, 1, 1)
    #
    # def test_csv_to_data_mapping_is_correct(self):
    #     #todo
    #     #VERY LIKELY YOU WILL HAVE TO CHANGE TEAMS IDs to match data preprocessor
    #     first_row = self.loader.get_events()[0]
    #     fr = first_row
    #     assert str(fr.event_id) == '170089'
    #     assert fr.league == 'England: Premier League'
    #     assert fr.event_date == datetime(2005,1,2)
    #     assert fr.home_team == 'Fulham'
    #     assert fr.away_team == 'Crystal Palace'
    #     assert fr.away_score == 1
    #     assert fr.home_score == 3
    #     assert fr.home_odds == 1.9456
    #     assert fr.draw_odds == 3.2333
    #     assert fr.away_odds == 3.6722



