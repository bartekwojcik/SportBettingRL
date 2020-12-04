from exp.experiment_main import experiment_main
from exp.algorithm_registry import AlgorithmRegistry
from exp.utils.misc import str2bool
import argparse


def parse():
    BARTEKS_DOCUMENTS =  r"RESULTS"
    BARTEKS_KERAS_PATH = r'resources\keras_model\model.h5'
    BARTEKS_DATA = R'resources\closing_odds_trimmed.csv'

    allowed_algorithms = AlgorithmRegistry.get_algorithms_map().keys()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path", type=str, default=BARTEKS_DOCUMENTS, help='Folder where results will be saved'
    )
    parser.add_argument(
        "--data_path", type=str, default=BARTEKS_DATA, help='path to CSV file with data'
    )
    parser.add_argument(
        "--model_path", type=str, default=BARTEKS_KERAS_PATH,help='path to keras model for winner prediction'
    )
    parser.add_argument(
        "--video_record_test", type=str2bool, default=True, help='select for video recording when testing. Environment must have render function implemented'
    )
    parser.add_argument(
        "--render_test", type=str2bool, default=True,help='select to render when testing. Environment must have render function implemented'
    )
    parser.add_argument("--validate_env_render", type=str2bool, default=True)
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1000,
        help="total timesteps for training. I think it has to be multiplied by number of environments in DummyVecEnv",
    )
    parser.add_argument(
        "--episode_max_steps",
        type=int,
        help="Can be None. For instance when original env already has time wrapper",
    )
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument(
        "--verbose", type=str2bool, default=True, help="verbose training algorithm"
    )

    parser.add_argument(
        "--norm_obs",
        type=str2bool,
        default=False,
        help="add observation normalisation wrapper",
    )
    parser.add_argument(
        "--norm_reward",
        type=str2bool,
        default=False,
        help="add reward normalisation wrapper",
    )
    parser.add_argument(
        "--norm_actions",
        type=str2bool,
        default=False,
        help="add [-1,1] actions normalisation wrapper for contious actions",
    )
    parser.add_argument(
        "--frame_stack",
        type=str2bool,
        default=False,
        help="add 4 past observations as an input when training",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=allowed_algorithms,
        type=str,
        help="coma seperated names of desired algorithms to test",
        #default=["A2C",'DQN', 'PPO'],
        default=['PPO'],
    )
    #DDPG - ValueError: Error: the model does not support multiple envs; it requires a single vectorized environment.
    #TD3 - ValueError: Error: the model does not support multiple envs; it requires a single vectorized environment.

    parser.add_argument("--log", type=str2bool, default=True, help='when debuging you might not want to log things to neptune.ai')
    parser.add_argument("--neptune_api_token", type=str, help='your API key for neptune ai. It\'s free and neat. https://neptune.ai/')

    args = parser.parse_args()
    return args


def go(args):
    save_path = args.save_path
    video_record_test = args.video_record_test
    total_timesteps = args.total_timesteps
    n_eval_episodes = args.n_eval_episodes
    verbose = args.verbose
    render_test = args.render_test
    algorithms = args.algorithms
    validate_env_render = args.validate_env_render
    log = args.log
    neptune_api_token = args.neptune_api_token
    norm_obs = args.norm_obs
    norm_reward = args.norm_reward
    episode_max_steps = args.episode_max_steps
    norm_actions = args.norm_actions
    frame_stack = args.frame_stack
    data_path = args.data_path
    model_path = args.model_path

    if neptune_api_token is None:
        print("since neptune api token is None i can't log much")
        log = False

    experiment_main(
        root_save_path=save_path,
        video_record_test=video_record_test,
        total_timesteps=total_timesteps,
        n_eval_episodes=n_eval_episodes,
        verbose=verbose,
        render_test=render_test,
        algorithms=algorithms,
        validate_env_render=validate_env_render,
        log=log,
        neptune_api_token=neptune_api_token,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        episode_max_steps=episode_max_steps,
        norm_actions=norm_actions,
        frame_stack=frame_stack,
        path_to_data=data_path,
        path_to_keras_model=model_path,
    )


if __name__ == "__main__":
    args = parse()
    go(args)
