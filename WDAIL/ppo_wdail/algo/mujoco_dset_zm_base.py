'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np


class Dset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, subsample_frequency=20, randomize=True):
        traj_data = np.load(expert_path, allow_pickle=True)

        rets_= traj_data['ep_rets']
        index = np.argsort(rets_)
        # index_ = np.random.choice(len(rets_),traj_limitation)

        # aa = len(traj_data['obs'])
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        # obs = traj_data['obs'][:traj_limitation]
        # acs = traj_data['acs'][:traj_limitation]
        select_bset = False
        if select_bset:
            sel_idx = index[::-1][:traj_limitation]
        else:
            sel_idx = np.random.choice(len(rets_),traj_limitation)
            # sel_idx = [0: traj_limitation]

        # obs = traj_data['obs'][sel_idx]
        # acs = traj_data['acs'][sel_idx]

        obs = traj_data['obs'][:traj_limitation]
        acs = traj_data['acs'][:traj_limitation]

        obs, acs = self.subsample(obs, acs, subsample_frequency)

        def flatten(x):
            # x.shape = (E,), or (E, L, D)
            _, size = x[0].shape
            episode_length = [len(i) for i in x]
            y = np.zeros((sum(episode_length), size))
            start_idx = 0
            for l, x_i in zip(episode_length, x):
                y[start_idx:(start_idx+l)] = x_i
                start_idx += l
                return y
        self.obs = np.array(flatten(obs))
        self.acs = np.array(flatten(acs))
        self.rets = traj_data['ep_rets'][:traj_limitation]
        # self.rets = traj_data['ep_rets'][sel_idx]

        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.randomize)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                              self.acs[:int(self.num_transition*train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                            self.acs[int(self.num_transition*train_fraction):, :],
                            self.randomize)
        self.log_info()

    def subsample(self, obs, acts, subsample_frequency=20):

        num_trajectories = len(obs)
        start_idx = np.random.randint(0, subsample_frequency, num_trajectories)
        t_obs = []
        t_acts = []

        for i in range(num_trajectories):
            obs_i = obs[i]
            acts_i = acts[i]

            t_obs.append(obs_i[start_idx[i]::subsample_frequency])
            t_acts.append(acts_i[start_idx[i]::subsample_frequency])

            # t_obs.append(obs[i, start_idx[i]::subsample_frequency])
            # t_acts.append(acts[i, start_idx[i]::subsample_frequency])

        t_obs = np.array(t_obs)
        t_acts = np.array(t_acts)


        return t_obs, t_acts


    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)
        logger.log("Average returns: %f" % self.avg_ret)
        logger.log("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError


    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
