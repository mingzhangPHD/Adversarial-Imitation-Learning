"""The main framwork for this work.
See README for usage.
"""

import argparse
import torch

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# set key parameters
def argsparser():
    parser = argparse.ArgumentParser("WDAIL")
    parser.add_argument('--env_name', help='environment ID', default='Walker2d-v2')
    parser.add_argument('--algo', help='algorithm ID', default='WDAIL')
    parser.add_argument('--log-dir', default='/tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
    # general
    parser.add_argument('--total_steps', help='total steps', type=int, default=10e6)
    parser.add_argument('--num_env_steps', help='total steps', type=int, default=10e6)
    parser.add_argument('--evaluate_every', help='evaluate every', type=int, default=2e10)
    parser.add_argument('--save_condition', help='save_condition', type=int, default=1000)
    parser.add_argument('--num_model', help='num_model', type=int, default=10)
    parser.add_argument('--use_device', help='use_device', type=bool, default=True)
    parser.add_argument('--cuda', help='num_model', type=int, default=0)
    parser.add_argument('--seed', help='seed', type=int, default=1)
    parser.add_argument('--use_linear_lr_decay', help='use linear lr decay', type=bool, default=True)
    parser.add_argument('--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')

    #ppo
    parser.add_argument('--num_processes', help='num_processes', type=int, default=1)
    parser.add_argument('--nsteps', help='nsteps', type=int, default=2048)
    parser.add_argument('--lr', help='learning rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--ppo_epoch', help='ppo epoch num', type=int, default=10)
    parser.add_argument('--hidden_size', help='hidden size', type=int, default=64)
    parser.add_argument('--ppo_entcoeff', help='entropy coefficiency of policy', type=float, default=1e-3) #default=1e-3
    parser.add_argument('--ppo_obs_norm', help='ppo_vec_norm', type=bool, default=True)
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=True, help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.00, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')

    # #gail
    parser.add_argument('--gail', help='if gail', type=bool, default=True)
    # parser.add_argument('--expert_path', help='trajs path', type=str, default='../data/baseline/deterministic.trpo.HalfCheetah.0.00.npz')
    parser.add_argument('--expert_path', help='trajs path', type=str, default='../data/ikostirkov/trajs_walker.h5')
    parser.add_argument('--gail-experts-dir', default='./gail_experts', help='directory that contains expert demonstrations for gail')
    parser.add_argument('--gail_batch_size', type=int, default=128, help='gail batch size (default: 128)')
    parser.add_argument('--gail_thre', help='number of steps to train discriminator in each epoch', type=int, default=10)
    parser.add_argument('--gail_pre_epoch', help='number of steps to train discriminator in each epoch', type=int, default=100)
    parser.add_argument('--gail_epoch', help='number of steps to train discriminator in each epoch', type=int, default=5)
    parser.add_argument('--num_trajs', help='num trajs', type=int, default=4)
    parser.add_argument('--subsample_frequency', help='num trajs', type=int, default=1)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    parser.add_argument('--use-proper-time-limits', action='store_true', default=True, help='compute returns taking into account time limits')
    parser.add_argument('--log-interval', type=int, default=1, help='log interval, one log per n updates (default: 10)')

    parser.add_argument('--reward_type', type=int, default=0, help='0,1,2,3,4')
    parser.add_argument('--update_rms', type=bool, default=False, help='False or True')

    return parser.parse_args()

def train(args):

    # from ppo_gail_iko.algo.ppo4multienvs import PPO, ReplayBuffer
    from ppo_wdail.algo.ppo import PPO
    from ppo_wdail.tools.storage import RolloutStorage
    from ppo_wdail.tools.model import Policy

    from ppo_wdail.algo.wdgail import Discriminator, ExpertDataset
    from ppo_wdail.algo.mujoco_dset_zm_iko import Mujoco_Dset

    from ppo_wdail.tools.learn import gailLearning_mujoco, Learning_process, Learning_process_record, gailLearning_mujoco_test, gailLearning_mujoco_origin
    from ppo_wdail.tools.envs import make_vec_envs

    from ppo_wdail.tools import utli
    from ppo_wdail.tools import utils

    from collections import deque
    import time
    import numpy as np


    # from nets.network import ActorCritic_mujoco as ActorCritic
    cl_args = args
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)

    device = torch.device('cuda:'+ str(cl_args.cuda) if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    # envs_eval = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                      args.gamma, args.log_dir, device, False)
    envs_eval = []

    # network
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    # discriminator
    # discr = Discriminator(envs.observation_space.shape[0] + envs.action_space.shape[0], 100, device)
    discr = Discriminator(envs.observation_space.shape[0] + envs.action_space.shape[0], 100, device, args.reward_type, args.update_rms)

    file_name = os.path.join(
        args.gail_experts_dir, "trajs_{}.pt".format(
            args.env_name.split('-')[0].lower()))

    gail_train_loader = torch.utils.data.DataLoader(
        ExpertDataset(
        file_name, num_trajectories=args.num_trajs, subsample_frequency=args.subsample_frequency),
        batch_size=args.gail_batch_size,
        shuffle=True,
        drop_last=True)

    # The buffer
    rollouts = RolloutStorage(args.nsteps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    # The buffer for the expert -> refer to dataset/mujoco_dset.py
    # expert_path = cl_args.expert_path+cl_args.env_id+".h5"
    # expert_buffer = Mujoco_Dset(cl_args.expert_path, traj_limitation=cl_args.num_trajs, subsample_frequency=args.subsample_frequency)

    model = gailLearning_mujoco_origin(cl_args=cl_args,
                                       envs=envs,
                                       envs_eval=envs_eval,
                                       actor_critic=actor_critic,
                                       agent=agent,
                                       discriminator=discr,
                                       rollouts=rollouts,
                                       gail_train_loader=gail_train_loader,
                                       device=device,
                                       utli=utli)

    return 0


def main(args):

    model, env = train(args)

    # if args.play:
    #
    #     obs = env.reset()
    #
    #     state = model.initial_state if hasattr(model, 'initial_state') else None
    #     dones = np.zeros((1,))
    #
    #     episode_rew = 0
    #     while True:
    #         if state is not None:
    #             actions, _, state, _ = model.step(obs,S=state, M=dones)
    #         else:
    #             actions, _, _, _ = model.step(obs)
    #
    #         obs, rew, done, _ = env.step(actions)
    #         episode_rew += rew[0] if isinstance(env, VecEnv) else rew
    #         env.render()
    #         done = done.any() if isinstance(done, np.ndarray) else done
    #         if done:
    #             print(f'episode_rew={episode_rew}')
    #             episode_rew = 0
    #             obs = env.reset()
    #
    #     env.close()

    return model

if __name__ == '__main__':

    args = argsparser()
    main(args)

