import unittest
import pandas as pd
import os
from datetime import datetime
from betting_env.data_preprocessor import DataPreprocessor
from betting_env.utils import label_encoder_to_dictionary

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self) -> None:
        PATH_TO_DATA = os.path.join(".", "resources", "kaggle_example_betting_csv.txt")
        self.df = pd.read_csv(PATH_TO_DATA, sep='\t')
        self.preprocessor = DataPreprocessor()
        self.columns_to_encode = ['league', 'home_team', 'away_team']
        #todo might want to use CsvColumnsMappings just like in other tests


    def test_data_teams_are_correcly_encoded(self):

        (encoded_df, col_to_encoder_map ) = self.preprocessor.encode_columns(self.df, self.columns_to_encode)

        league_encoder = col_to_encoder_map['league']
        le_dict =label_encoder_to_dictionary(league_encoder)
        assert le_dict['England: Premier League'] == 0
        assert le_dict['Suckland: Premier League'] == 1

        ht_encoder = col_to_encoder_map['home_team']
        ht_dict =label_encoder_to_dictionary(ht_encoder)

        assert ht_dict['Aston Villa'] == 0
        assert ht_dict['Bolton'] == 1
        assert ht_dict['Charlton'] == 2
        assert ht_dict['Fulham'] == 3
        assert ht_dict['Liverpool'] == 4

        at_encoder = col_to_encoder_map['away_team']
        at_dict =label_encoder_to_dictionary(at_encoder)

        assert at_dict['Arsenal'] == 0
        assert at_dict['Blackburn'] == 1
        assert at_dict['Chelsea'] == 2
        assert at_dict['Crystal Palace'] == 3
        assert at_dict['West Brom'] == 4

    def test_that_encoded_df_is_the_same_as_dencoded(self):
        (encoded_df, col_to_encoder_map) = self.preprocessor.encode_columns(self.df, self.columns_to_encode)

        assert len(col_to_encoder_map) == len(self.columns_to_encode)
        assert len(self.columns_to_encode) > 0

        for column_name in self.columns_to_encode:
            encoder = col_to_encoder_map[column_name]

            reverse_column = encoder.inverse_transform(encoded_df[column_name])
            assert len(reverse_column) == len(self.df[column_name])
            assert (reverse_column == self.df[column_name]).all()


    def test_df_is_in_the_same_order_as_input_df(self):

        (encoded_df, _ )  = self.preprocessor.encode_columns(self.df, self.columns_to_encode)

        column_to_compare = 'match_id'

        assert (encoded_df[column_to_compare] == self.df[column_to_compare]).all()








