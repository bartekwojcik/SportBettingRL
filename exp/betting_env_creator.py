from betting_env.data_loading_strategies.csv_loading_strategy import (
    CSVLoadingStrategy,
)
from betting_env.data_loading_strategies.csv_column_mapping import (
    CsvColumnsMappings,
)
from betting_env.data_preprocessor import DataPreprocessor
from betting_env.event_factory import EventFactory
from betting_env.environment.betting_environment import BettingEnv
from betting_env.reward_calculator import DecimalRewardCalculator
from betting_env.load_keras_model import load_model
from betting_env.environment.keras_wrapper import KerasWrapperEnv
from betting_env.environment.normalize_state_wrapper import NormalizeStateWrapper


def get_env_function(PATH_TO_MODEL:str,PATH_TO_DATA:str=None, ):
    """
    Creates and loads Betting environment.

    :param PATH_TO_MODEL: path to keras model
    :param PATH_TO_DATA: path to CSV file
    :return: Function that creates gym.Env, string name of this env for loggin, and dictionaries of parameters
    """


    PATH_TO_DATA = (
        r"C:\Users\kicjo\Documents\Repositories\RLExperiment\resources\closing_odds_trimmed.csv"
        if PATH_TO_DATA is None
        else PATH_TO_DATA
    )
    col_map = CsvColumnsMappings(
        event_id="match_id",
        event_date="match_date",
        away_team="away_team",
        home_team="home_team",
        league="league",
        away_score="away_score",
        home_score="home_score",
        away_odds="avg_odds_away_win",
        home_odds="avg_odds_home_win",
        draw_odds="avg_odds_draw",
        column_separator=",",
        date_format="%Y-%m-%d",
    )

    loading_strategy = CSVLoadingStrategy(PATH_TO_DATA, col_map)
    df = loading_strategy.dataframe

    columns_to_encode = ["league", "home_team", "away_team"]
    preprocessor = DataPreprocessor()
    encoded_df, col_to_encoder_map = preprocessor.encode_columns(df, columns_to_encode)
    event_factory = EventFactory(encoded_df.copy(), col_map)
    reward_calculator = DecimalRewardCalculator()

    _ = event_factory.get_all_events()
    print("loading data finished")
    keras_model = load_model(PATH_TO_MODEL)

    def make_env(bankroll=100, seed: int = 0,winning_limit=150):
        """
        This function should be possible to call without passing any arguments for sake of environment validation
        """
        env = BettingEnv(
            event_factory=event_factory,
            bankroll=bankroll,
            reward_calculator=reward_calculator,
            seed=seed,
            seed_end_range=seed+1000,
            winning_limit=winning_limit,

        )

        normalized_env = NormalizeStateWrapper(env=env)

        keras_wrapper = KerasWrapperEnv(normalized_env,keras_model)

        return keras_wrapper

    #they have to match paramters of make_env function
    train_env_parameters_dict = {"seed": 0, "bankroll": 100,'winning_limit':150}
    eval_env_parameters_dict = {"seed": 20000, "bankroll": 100,'winning_limit':150}
    test_env_parameters_dict = {"seed": 40000, "bankroll": 100,'winning_limit':150}

    return (
        make_env, #function that returns gym.Env instance
        "BettingEnv", #string name of this environment for logging purpouse
        train_env_parameters_dict, # dictionary of parameters to be passed to make_env function when creating train set
        eval_env_parameters_dict, # dictionary of parameters to be passed to make_env function when creating eval set
        test_env_parameters_dict,# dictionary of parameters to be passed to make_env function when creating test set
    )

