import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd

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

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, reward_type, update_rms, cliprew_down=-10.0, cliprew_up=10.0):
        super(Discriminator, self).__init__()
        self.cliprew_down = cliprew_down
        self.cliprew_up = cliprew_up
        self.device = device
        self.reward_type = reward_type
        self.update_rms = update_rms

        # self.trunk = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim), nn.Tanh(),
        #     nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        #     nn.Linear(hidden_dim, 1), nn.Tanh()).to(device)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update_zm(self, replay_buf, expert_buf, obsfilt=None, batch_size=128):
        self.train()
        obs = replay_buf.obs
        obs_batch = obs[:-1].view(-1, *obs.size()[2:])
        states = obs_batch.cpu().detach().numpy()

        # states = np.concatenate(states,axis=1)
        actions = replay_buf.actions
        actions_batch = actions.view(-1,actions.size(-1))
        actions = actions_batch.cpu().detach().numpy()

        policy_buf = Dset(inputs=states[0:len(actions)], labels=actions, randomize=True)

        loss = 0
        g_loss =0.0
        gp =0.0
        n = 0

        # loss = 0

        # Sample replay buffer
        policy_state, policy_action = policy_buf.get_next_batch(batch_size)
        policy_state = torch.FloatTensor(policy_state).to(self.device)
        policy_action = torch.FloatTensor(policy_action).to(self.device)
        temp=[policy_state, policy_action]
        policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))

        # Sample expert buffer
        expert_state, expert_action = expert_buf.get_next_batch(batch_size)
        expert_state = obsfilt(expert_state, update=False)
        expert_state = torch.FloatTensor(expert_state).to(self.device)
        expert_action = torch.FloatTensor(expert_action).to(self.device)
        expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))

        # expert_loss = F.binary_cross_entropy_with_logits(
        #     expert_d,
        #     torch.ones(expert_d.size()).to(self.device))
        # policy_loss = F.binary_cross_entropy_with_logits(
        #     policy_d,
        #     torch.zeros(policy_d.size()).to(self.device))

        # expert_loss = torch.mean(expert_d).to(self.device)
        # policy_loss = torch.mean(policy_d).to(self.device)

        expert_loss = torch.mean(torch.tanh(expert_d)).to(self.device)
        policy_loss = torch.mean(torch.tanh(policy_d)).to(self.device)

        # gail_loss = expert_loss + policy_loss
        wd = expert_loss - policy_loss
        grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                         policy_state, policy_action)

        # loss += (gail_loss + grad_pen).item()
        loss += (-wd + grad_pen).item()
        g_loss += (wd).item()
        gp += (grad_pen).item()
        n += 1

        self.optimizer.zero_grad()
        # (gail_loss + grad_pen).backward()
        (-wd + grad_pen).backward()
        self.optimizer.step()


        return g_loss/n, gp/n, 0.0, loss / n

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        g_loss =0.0
        gp =0.0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            # expert_loss = F.binary_cross_entropy_with_logits(
            #     expert_d,
            #     torch.ones(expert_d.size()).to(self.device))
            # policy_loss = F.binary_cross_entropy_with_logits(
            #     policy_d,
            #     torch.zeros(policy_d.size()).to(self.device))

            # expert_loss = torch.mean(expert_d).to(self.device)
            # policy_loss = torch.mean(policy_d).to(self.device)

            expert_loss = torch.mean(torch.tanh(expert_d)).to(self.device)
            policy_loss = torch.mean(torch.tanh(policy_d)).to(self.device)

            # gail_loss = expert_loss + policy_loss
            wd = expert_loss - policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            # loss += (gail_loss + grad_pen).item()
            loss += (-wd + grad_pen).item()
            g_loss += (wd).item()
            gp += (grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            # (gail_loss + grad_pen).backward()
            (-wd + grad_pen).backward()
            self.optimizer.step()

        return g_loss/n, gp/n, 0.0, loss / n

    def update_origin(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        g_loss =0.0
        gp =0.0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            # expert_loss = F.binary_cross_entropy_with_logits(
            #     expert_d,
            #     torch.ones(expert_d.size()).to(self.device))
            # policy_loss = F.binary_cross_entropy_with_logits(
            #     policy_d,
            #     torch.zeros(policy_d.size()).to(self.device))

            expert_loss = torch.mean(expert_d).to(self.device)
            policy_loss = torch.mean(policy_d).to(self.device)

            # gail_loss = expert_loss + policy_loss
            wd = expert_loss - policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            # loss += (gail_loss + grad_pen).item()
            loss += (-wd + grad_pen).item()
            g_loss += (wd).item()
            gp += (grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            # (gail_loss + grad_pen).backward()
            (-wd + grad_pen).backward()
            self.optimizer.step()

        return g_loss/n, gp/n, 0.0, loss / n

    def update_zm_origin(self, replay_buf, expert_buf, obsfilt=None, batch_size=128):
        self.train()
        obs = replay_buf.obs
        obs_batch = obs[:-1].view(-1, *obs.size()[2:])
        states = obs_batch.cpu().detach().numpy()

        # states = np.concatenate(states,axis=1)
        actions = replay_buf.actions
        actions_batch = actions.view(-1,actions.size(-1))
        actions = actions_batch.cpu().detach().numpy()

        policy_buf = Dset(inputs=states[0:len(actions)], labels=actions, randomize=True)

        # loss = 0

        # Sample replay buffer
        policy_state, policy_action = policy_buf.get_next_batch(batch_size)
        policy_state = torch.FloatTensor(policy_state).to(self.device)
        policy_action = torch.FloatTensor(policy_action).to(self.device)
        temp=[policy_state, policy_action]
        policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))

        # Sample expert buffer
        expert_state, expert_action = expert_buf.get_next_batch(batch_size)
        expert_state = obsfilt(expert_state, update=False)
        expert_state = torch.FloatTensor(expert_state).to(self.device)
        expert_action = torch.FloatTensor(expert_action).to(self.device)
        expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_d,
            torch.ones(expert_d.size()).to(self.device))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_d,
            torch.zeros(policy_d.size()).to(self.device))

        gail_loss = expert_loss + policy_loss
        grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                         policy_state, policy_action)
        # print("gail_loss = %s,    gp=%s" % (gail_loss.item(), grad_pen.item()))

        loss = (gail_loss + grad_pen).item()
        # loss = (gail_loss).item()

        self.optimizer.zero_grad()
        (gail_loss + grad_pen).backward()
        # (gail_loss).backward()
        self.optimizer.step()

        return gail_loss.item(), grad_pen.item(), 0.0, loss

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            if self.reward_type == 0:
                s = torch.exp(d)
                reward = s
            elif self.reward_type == 1:
                s = torch.sigmoid(d)
                reward = - (1 - s).log()
            elif self.reward_type == 2:
                s = torch.sigmoid(d)
                reward = s
            elif self.reward_type == 3:
                s = torch.sigmoid(d)
                reward = s.exp()
            elif self.reward_type == 4:
                reward = d

            # s = torch.exp(d)
            # # reward = s.log() - (1 - s).log()
            # s = torch.sigmoid(d)
            # reward = s
            # # reward = d
            if self.returns is None:
                self.returns = reward.clone()

            if self.update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())
                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            else:
                return reward


            # ttt = torch.clamp(reward / np.sqrt(self.ret_rms.var[0] + 1e-8), self.cliprew_down, self.cliprew_up)
            # return torch.clamp(reward / np.sqrt(self.ret_rms.var[0] + 1e-8), self.cliprew_down, self.cliprew_up)
            # return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            # return torch.clamp(reward,self.cliprew_down, self.cliprew_up)
            # return reward

    def predict_reward_exp(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.exp(d)
            # s = torch.sigmoid(d)
            # reward = s.log() - (1 - s).log()
            reward = s
            # reward = d
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())
            # ttt = torch.clamp(reward / np.sqrt(self.ret_rms.var[0] + 1e-8), self.cliprew_down, self.cliprew_up)
            # return torch.clamp(reward / np.sqrt(self.ret_rms.var[0] + 1e-8), self.cliprew_down, self.cliprew_up)
            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            # return torch.clamp(reward,self.cliprew_down, self.cliprew_up)
            # return reward

    def predict_reward_t1(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            # s = torch.exp(d)
            s = torch.sigmoid(d)
            # reward = s.log() - (1 - s).log()
            reward = - (1 - s).log()
            # reward = d
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())
            # ttt = torch.clamp(reward / np.sqrt(self.ret_rms.var[0] + 1e-8), self.cliprew_down, self.cliprew_up)
            # return torch.clamp(reward / np.sqrt(self.ret_rms.var[0] + 1e-8), self.cliprew_down, self.cliprew_up)
            # return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            # return torch.clamp(reward,self.cliprew_down, self.cliprew_up)
            return reward

    def predict_reward_origin(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.exp(d)
            # s = torch.sigmoid(d)
            # reward = s.log() - (1 - s).log()
            # reward = - (1 - s).log()
            reward = s
            # reward = d
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())
            # ttt = torch.clamp(reward / np.sqrt(self.ret_rms.var[0] + 1e-8), self.cliprew_down, self.cliprew_up)
            # return torch.clamp(reward / np.sqrt(self.ret_rms.var[0] + 1e-8), self.cliprew_down, self.cliprew_up)
            # return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            # return torch.clamp(reward,self.cliprew_down, self.cliprew_up)
            return reward

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20):
        all_trajectories = torch.load(file_name)

        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}

        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories,)).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}

        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []

        for j in range(self.length):

            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1

    def __len__(self):
        return self.length

    def __getitem__(self, j):
        traj_idx, i = self.get_idx[j]

        return self.trajectories['states'][traj_idx][i], self.trajectories[
            'actions'][traj_idx][i]
