from typing import Tuple, Callable, Dict, Any

from exp.wrappers.time_limit_wrapper import TimeLimitWrapper
from stable_baselines.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    VecVideoRecorder,
    VecEnv,
    VecFrameStack,
    SubprocVecEnv,
)

from exp.wrappers.norm_action_space_wrapper import NormalizeActionWrapper


class TestTrainEnvCreator:
    def _setup_vec_env(
        self,
        env_creating_function: Callable,
        norm_actions: bool,
        env_parameters_dict,
        episode_max_steps: int = None,
    ):
        def _init():
            train_env = env_creating_function(**env_parameters_dict)
            if episode_max_steps is not None and isinstance(episode_max_steps, int):
                train_env = TimeLimitWrapper(train_env, max_steps=episode_max_steps)

            if norm_actions:
                train_env = NormalizeActionWrapper(train_env)

            return train_env

        return _init

    def _create_train_env(
        self,
        env_creating_function: Callable,
        norm_obs: bool,
        norm_reward: bool,
        norm_actions: bool,
        frame_stack: bool,
        env_parameters_dict,
        n_env: int = 8,
        eval: bool = False,
        episode_max_steps: int = None,
    ) -> VecEnv:

        train_env = DummyVecEnv(
            [
                self._setup_vec_env(
                    env_creating_function=env_creating_function,
                    norm_actions=norm_actions,
                    episode_max_steps=episode_max_steps,
                    env_parameters_dict=env_parameters_dict,
                )
                for _ in range(n_env)
            ]
        )

        if frame_stack:
            train_env = VecFrameStack(train_env, 4)

        if norm_obs or norm_reward:
            _norm_reward = False if eval else norm_reward
            # https://github.com/hill-a/stable-baselines/issues/820
            train_env = VecNormalize(
                train_env, norm_obs=norm_obs, norm_reward=_norm_reward
            )
        return train_env

    def _create_test_env(
        self,
        env_creating_function: Callable,
        episode_max_steps: int,
        norm_actions: bool,
        frame_stack: bool,
        env_parameters_dict,
        video_recording_path: str = None,
        video_length: int = 1000,
        n_env: int = 1,
    ) -> VecEnv:

        test_env = DummyVecEnv(
            [
                self._setup_vec_env(
                    env_creating_function=env_creating_function,
                    norm_actions=norm_actions,
                    episode_max_steps=episode_max_steps,
                    env_parameters_dict=env_parameters_dict,
                )
                for _ in range(n_env)
            ]
        )
        if frame_stack:
            test_env = VecFrameStack(test_env, 4)

        if video_recording_path is not None:
            test_env = VecVideoRecorder(
                test_env,
                video_recording_path,
                record_video_trigger=lambda x: x == 0,
                video_length=video_length,
            )

        return test_env

    def get_baselines_environment(
        self,
        env_creating_function: Callable,
        norm_obs: bool,
        norm_reward: bool,
        norm_actions: bool,
        frame_stack: bool,
        episode_max_steps: int,
        train_env_parameters_dict: Dict[str, Any],
        eval_env_parameters_dict: Dict[str, Any],
        test_env_parameters_dict: Dict[str, Any],
        video_recording_path: str = None,
        n_train_env: int = 8,
    ) -> Tuple[VecEnv, VecEnv, VecEnv]:

        train_env = self._create_train_env(
            env_creating_function=env_creating_function,
            episode_max_steps=episode_max_steps,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            frame_stack=frame_stack,
            norm_actions=norm_actions,
            env_parameters_dict=train_env_parameters_dict,
            n_env=n_train_env,
        )

        eval_env = self._create_train_env(
            env_creating_function=env_creating_function,
            episode_max_steps=episode_max_steps,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            n_env=1,
            frame_stack=frame_stack,
            norm_actions=norm_actions,
            env_parameters_dict=eval_env_parameters_dict,
        )

        test_env = self._create_test_env(
            env_creating_function=env_creating_function,
            episode_max_steps=episode_max_steps,
            video_recording_path=video_recording_path,
            n_env=1,
            frame_stack=frame_stack,
            norm_actions=norm_actions,
            env_parameters_dict=test_env_parameters_dict,
        )

        return train_env, test_env, eval_env
