from pathlib2 import Path

dac_dir = Path(__file__).parent
trajector_dir = dac_dir / 'trajs'
project_dir = dac_dir.parent.parent
# results_dir = project_dir / 'results'

results_dir = dac_dir/ 'results'

trained_model_dir = dac_dir/ 'trained_model'

trained_model_dir_rela = './trained_model'


if not trajector_dir.is_dir():
    trajector_dir.mkdir()

if not results_dir.is_dir():
    results_dir.mkdir()

if not trained_model_dir.is_dir():
    trained_model_dir.mkdir()

import argparse
# set key parameters
def ppo_argsparser():
    parser = argparse.ArgumentParser("PPO")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--algo_id', help='algorithm ID', default='PPO')
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--nenv', help='num env', type=int, default=1)
    parser.add_argument('--ppo_epoch', help='ppo epoch num', type=int, default=10)
    parser.add_argument('--hidden_size', help='hidden size', type=int, default=64)
    parser.add_argument('--nsteps', help='nsteps', type=int, default=2048)
    parser.add_argument('--total_steps', help='total steps', type=int, default=10e6)
    parser.add_argument('--learning_rate', help='learning rate', type=int, default=3e-4)
    parser.add_argument('--evaluate_every', help='evaluate every', type=int, default=20480)
    parser.add_argument('--save_condition', help='save_condition', type=int, default=1000)
    parser.add_argument('--num_model', help='num_model', type=int, default=10)
    parser.add_argument('--use_device', help='use_device', type=bool, default=True)
    parser.add_argument('--cuda', help='num_model', type=int, default=0)

    return parser.parse_args()