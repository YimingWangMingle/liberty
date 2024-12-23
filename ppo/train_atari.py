from utils.env_wrapper.create_env import create_multiple_envs
from utils.seeds.seeds import set_seeds
import os
import numpy as np
import torch
from models_atari import net, bisim_net
from datetime import datetime
import copy
import os
import argparse
from collections import deque

import torch
from torch.distributions.categorical import Categorical

def select_actions(pi, deterministic=False):
    cate_dist = Categorical(pi)
    if deterministic:
        return torch.argmax(pi, dim=1).item()
    else:
        return cate_dist.sample().unsqueeze(-1)

def evaluate_actions(pi, actions):
    cate_dist = Categorical(pi)
    return cate_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1), cate_dist.entropy().mean()

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1.-done)
        discounted.append(r)
    return discounted[::-1]


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--env-name', type=str, default='BreakoutNoFrameskip-v4', help='the environment name')
    parse.add_argument('--lr', type=float, default=7e-4, help='learning rate of the algorithm')
    parse.add_argument('--value-loss-coef', type=float, default=0.5, help='the coefficient of value loss')
    parse.add_argument('--tau', type=float, default=0.95, help='gae coefficient')
    parse.add_argument('--cuda', action='store_true', help='use cuda do the training')
    parse.add_argument('--transition_model_type', type=str, default='probabilistic', help='the transition model')
    parse.add_argument('--total-frames', type=int, default=50000000, help='the total frames for training')
    parse.add_argument('--eps', type=float, default=1e-5, help='param for adam optimizer')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--nsteps', type=int, default=5, help='the steps to update the network')
    parse.add_argument('--num-workers', type=int, default=16, help='the number of cpu you use')
    parse.add_argument('--entropy-coef', type=float, default=0.01, help='entropy-reg')
    parse.add_argument('--log-interval', type=int, default=100, help='the log interval')
    parse.add_argument('--alpha', type=float, default=0.99, help='the alpha coe of RMSprop')
    parse.add_argument('--max-grad-norm', type=float, default=0.5, help='the grad clip')
    parse.add_argument('--use-gae', action='store_true', help='use-gae')
    parse.add_argument('--log-dir', type=str, default='logs/', help='log dir')
    parse.add_argument('--env-type', type=str, default='atari', help='the type of the environment')
    parse.add_argument('--lr-decay', action='store_true', help='use lr-decay')
    parse.add_argument('--r-ext-coef', type=float, default=1, help='the extrnisc reward coef')
    parse.add_argument('--r-in-coef', type=float, default=0.01, help='the intrinsic reward coef')
    parse.add_argument('--vloss-coef', type=float, default=0.1)

    args = parse.parse_args()

    return args


class ppo_agent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args
        self.net = net(self.envs.action_space.n)
        self.intrinsic_net = bisim_net()
        if self.args.cuda:
            self.net.cuda()
            self.intrinsic_net.cuda()
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.args.lr, eps=self.args.eps, alpha=self.args.alpha)
        self.intrinsic_optimizer = torch.optim.RMSprop(self.intrinsic_net.parameters(), lr=self.args.lr, eps=self.args.eps, alpha=self.args.alpha)
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir + self.args.env_name + '/'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps,) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers,) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        self.obs[:] = self.envs.reset()
        self.dones = [False for _ in range(self.args.num_workers)]
        self.state_optims = None

    def learn(self):
        num_updates = self.args.total_frames // (self.args.num_workers * self.args.nsteps)
        episode_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        final_rewards = np.zeros((self.args.num_workers, ), dtype=np.float32)
        episode_rewards_stat = deque(maxlen=10)
        for update in range(num_updates):
            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)
            mb_obs, mb_rewards_ex, mb_actions, mb_dones, mb_obs_, mb_v_ex, mb_v_mix = [], [], [], [], [], [], []
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    input_tensor = self._get_tensors(self.obs)
                    v_mix, pi = self.net(input_tensor)
                    _, v_ex = self.intrinsic_net(input_tensor)
                actions = select_actions(pi)
                cpu_actions = actions.squeeze(1).cpu().numpy()
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(cpu_actions)
                mb_dones.append(self.dones)
                mb_v_ex.append(v_ex.detach().cpu().numpy().squeeze())
                mb_v_mix.append(v_mix.detach().cpu().numpy().squeeze())
                obs_, rewards, dones, infos = self.envs.step(cpu_actions)
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards_stat.append(info['episode']['r'])
                mb_obs_.append(np.copy(obs_))
                self.dones = dones
                mb_rewards_ex.append(rewards)
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n]*0
                self.obs = obs_
                episode_rewards += rewards
                masks = np.array([0.0 if done else 1.0 for done in dones], dtype=np.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
            mb_dones.append(self.dones)
            mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
            mb_obs_ = np.asarray(mb_obs_, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
            mb_rewards_in = self._compute_intrinsic_rewards(mb_obs, mb_obs_)
            mb_rewards_in = mb_rewards_in.reshape((self.args.num_workers, self.args.nsteps))
            mb_rewards_ex = np.asarray(mb_rewards_ex, dtype=np.float32).swapaxes(1, 0)
            mb_rewards_mix = self.args.r_ext_coef * mb_rewards_ex + self.args.r_in_coef * mb_rewards_in
            mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
            mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
            mb_v_ex = np.asarray(mb_v_ex, dtype=np.float32).swapaxes(1, 0)
            mb_v_mix = np.asarray(mb_v_mix, dtype=np.float32).swapaxes(1, 0)
            mb_masks = mb_dones[:, :-1]
            mb_dones = mb_dones[:, 1:]
            with torch.no_grad():
                input_tensor = self._get_tensors(self.obs)
                last_values_mix, _ = self.net(input_tensor)
                last_values_mix = last_values_mix.detach().cpu().numpy().squeeze()
                _, last_values_ex = self.intrinsic_net(input_tensor)
                last_values_ex = last_values_ex.detach().cpu().numpy().squeeze()
            mb_returns_ex, mb_returns_mix = np.zeros(mb_rewards_ex.shape), np.zeros(mb_rewards_in.shape)
            for n, (rewards_ex, rewards_mix, dones, value_mix, value_ex) in enumerate(zip(mb_rewards_ex, mb_rewards_mix, mb_dones, last_values_mix, last_values_ex)):
                rewards_ex = rewards_ex.tolist()
                rewards_mix = rewards_mix.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    returns_ex = discount_with_dones(rewards_ex+[value_ex], dones+[0], self.args.gamma)[:-1]
                    returns_mix = discount_with_dones(rewards_mix+[value_mix], dones+[0], self.args.gamma)[:-1]
                else:
                    returns_ex = discount_with_dones(rewards_ex, dones, self.args.gamma)
                    returns_mix = discount_with_dones(rewards_mix, dones, self.args.gamma)
                mb_returns_ex[n] = returns_ex
                mb_returns_mix[n] = returns_mix
            mb_rewards_ex = mb_rewards_ex.flatten()
            mb_rewards_in = mb_rewards_in.flatten()
            mb_returns_ex = mb_returns_ex.flatten()
            mb_returns_mix = mb_returns_mix.flatten()
            mb_actions = mb_actions.flatten()
            mb_v_ex = mb_v_ex.flatten()
            mb_v_mix = mb_v_mix.flatten()
            mb_dones = mb_dones.flatten()
            mb_masks = mb_masks.flatten()
            dis_v_mix_last = np.zeros([mb_obs.shape[0]], np.float32)
            coef_mat = np.zeros([mb_obs.shape[0], mb_obs.shape[0]], np.float32)
            for i in range(mb_obs.shape[0]):
                dis_v_mix_last[i] = self.args.gamma ** (self.args.nsteps - i % self.args.nsteps) * last_values_mix[i // self.args.nsteps]
                coef = 1.0
                for j in range(i, mb_obs.shape[0]):
                    if j > i and j % self.args.nsteps == 0:
                        break
                    coef_mat[i][j] = coef
                    coef *= self.args.gamma
                    if mb_dones[j]:
                        dis_v_mix_last[i] = 0
                        break
            vl, al, ent = self._update_network(mb_obs, mb_obs_, mb_masks, mb_actions, mb_rewards_ex, mb_returns_ex, mb_v_ex, mb_v_mix, \
                                                dis_v_mix_last, coef_mat)
            if update % self.args.log_interval == 0:
                print('[{}] Update: {}/{}, Frames: {}, Rewards: {:.1f}, VL: {:.3f}, PL: {:.3f}, Ent: {:.2f}, Min: {}, Max:{}, R_in: {:.3f}'.format(\
                    datetime.now(), update, num_updates, (update+1)*(self.args.num_workers * self.args.nsteps),\
                    final_rewards.mean(), vl, al, ent, final_rewards.min(), final_rewards.max(), np.mean(mb_rewards_in)))
                torch.save(self.net.state_dict(), self.model_path + 'model.pt')
    
    def _update_network(self, obs, obs_, masks, actions, r_ex, returns_ex, v_ex, v_mix, dis_v_mix_last, coef_mat):
        r_ex_tensor = torch.tensor(r_ex, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        dis_v_mix_last_tensor = torch.tensor(dis_v_mix_last, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        coef_mat_tensor = torch.tensor(coef_mat, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        v_mix_tensor = torch.tensor(v_mix, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        r_in_tensor = self._compute_intrinsic_rewards(obs, obs_, requires_grad=True)
        r_mix_tensor = self.args.r_ext_coef * r_ex_tensor + self.args.r_in_coef * r_in_tensor
        returns_mix_tensor = torch.matmul(coef_mat_tensor, r_mix_tensor) + dis_v_mix_last_tensor
        adv_mix_tensor = returns_mix_tensor - v_mix_tensor
        input_tensor = self._get_tensors(obs)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device='cuda' if self.args.cuda else 'cpu').unsqueeze(1)
        values_mix, pi = self.net(input_tensor)
        action_log_probs, dist_entropy = evaluate_actions(pi, actions_tensor)
        value_loss = (values_mix - returns_mix_tensor.detach()).pow(2).mean()
        action_loss = -(adv_mix_tensor * action_log_probs).mean()
        total_loss = action_loss + self.args.value_loss_coef * value_loss - self.args.entropy_coef * dist_entropy

        self.optimizer.zero_grad()
        self.intrinsic_optimizer.zero_grad()
        grads = torch.autograd.grad(total_loss, self.net.parameters(), create_graph=True)
        net_new = copy.deepcopy(self.net)

        for (_, param), grad in zip(self.net.named_parameters(), grads):
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
                param.grad.data.copy_(grad.data)
            else:
                param.grad.data.copy_(grad.data)
        self.optimizer.step()
        if self.state_optims is None:
            self.state_optims = self.optimizer.state_dict()['state'].values()
            self.init_optim = True
        for (_, param), grad, state_optim in zip(net_new.named_parameters(), grads, self.state_optims):
            state_groups = self.optimizer.state_dict()['param_groups'][0]
            alpha = state_groups['alpha']
            eps = state_groups['eps']
            lr = state_groups['lr']
            if self.init_optim:
                square_avg = torch.zeros_like(param)
            else:
                square_avg = state_optim['square_avg'].clone()
            square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
            square_avg.add_(eps).sqrt()
            param.requires_grad = False
            param.addcdiv_(-lr, grad, square_avg)
        self.init_optim = False
        self.state_optims = self.optimizer.state_dict()['state'].values()
        adv_ex = returns_ex - v_ex
        adv_ex_tensor = torch.tensor(adv_ex, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        returns_ex_tensor = torch.tensor(returns_ex, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1) 
        _, pi_new = net_new(input_tensor)
        action_log_probs_new, _ = evaluate_actions(pi_new, actions_tensor)
        new_ratio = torch.exp(action_log_probs_new - action_log_probs.detach())
        in_policy_loss = -(new_ratio * adv_ex_tensor).mean()
        _, values_ex = self.intrinsic_net(input_tensor)
        in_value_loss = (returns_ex_tensor - values_ex).pow(2).mean()
        in_total_loss = in_policy_loss + self.args.vloss_coef * in_value_loss
        eta_grads = torch.autograd.grad(in_total_loss, self.intrinsic_net.parameters())
        for param, grad in zip(self.intrinsic_net.parameters(), eta_grads):
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
                param.grad.data.copy_(grad.data)
            else:
                param.grad.data.copy_(grad.data)
        self.intrinsic_optimizer.step()
        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def _compute_intrinsic_rewards(self, obs, obs_, requires_grad=False):

        obs_tensor = self._get_tensors(obs)
        obs_next_tensor = self._get_tensors(obs_)
        if not requires_grad:
            with torch.no_grad():
                feats_in, _ = self.intrinsic_net(obs_tensor)
                feats_in_next, _ = self.intrinsic_net(obs_next_tensor)
        else:
            feats_in, _ = self.intrinsic_net(obs_tensor)
            feats_in_next, _ = self.intrinsic_net(obs_next_tensor)
        feats_in = torch.nn.functional.normalize(feats_in, p=2, dim=1)
        feats_in_next = torch.nn.functional.normalize(feats_in_next, p=2, dim=1)
        feats_in = feats_in.unsqueeze(1)
        feats_in_next = feats_in_next.unsqueeze(1)
        feat_vec = torch.cat([feats_in, feats_in_next], dim=1)
        feat_vec_T = torch.transpose(feat_vec, 1, 2)
        kernel_mat = torch.matmul(feat_vec, feat_vec_T)        
        rewards_in = torch.det(kernel_mat)
        return rewards_in.unsqueeze(-1) if requires_grad else rewards_in.unsqueeze(-1).detach().cpu().numpy()
    
    def _get_tensors(self, obs):
        input_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        return input_tensor

    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr





if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    args = get_args()
    envs = create_multiple_envs(args)
    set_seeds(args)
    ppo_trainer = ppo_agent(envs, args)
    ppo_trainer.learn()
    envs.close()
