import gym
from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env.vec_normalize import VecNormalize
from bench.monitor import Monitor
import logger

def build_env(args, nenv, norm_env = True):

    def make_env():
        def _thunk():
            env = gym.make(args.env_id)
            env.seed(args.seed)    # to make the result more reproducibility
            env = Monitor(env, logger.get_dir(), allow_early_resets=True)
            return env
        return _thunk

    envs = [make_env() for i in range(nenv)]
    envs = SubprocVecEnv(envs)
    if norm_env:
        envs = VecNormalize(envs)

    return envs

def build_env4gail(args, nenv):

    def make_env():
        def _thunk():
            env = gym.make(args.env_id)
            env.seed(args.seed)    # to make the result more reproducibility
            env = Monitor(env, logger.get_dir(), allow_early_resets=True)
            return env
        return _thunk

    envs = [make_env() for i in range(nenv)]
    envs = SubprocVecEnv(envs)
    envs = VecNormalize(envs)

    return envs

def build_env4wdail(args, nenv):

    def make_env():
        def _thunk():
            env = gym.make(args.env_id)
            env.seed(args.seed)    # to make the result more reproducibility
            env = Monitor(env, logger.get_dir(), allow_early_resets=True)
            return env
        return _thunk

    envs = [make_env() for i in range(nenv)]
    envs = SubprocVecEnv(envs)
    if args.env_norm:
        envs = VecNormalize(envs)

    return envs