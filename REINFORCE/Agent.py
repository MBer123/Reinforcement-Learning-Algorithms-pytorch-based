import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import os
import random


# 确保实验的可重复性和结果的一致性
def set_seed(seed: int = 42) -> None:
    T.manual_seed(seed)
    T.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False


device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
set_seed()


class ActorNetwork(nn.Module):
    """
    :param lr: 学习率
    :param state_dim: 状态空间维度
    :param action_dim: 动作空间维度
    :param fc1_dim: 第一个全连接层的维度
    :param fc2_dim: 第二个全连接层的维度
    :param is_continuous: 是否为连续动作空间
    """
    def __init__(self, lr, state_dim, action_dim, fc1_dim, fc2_dim, is_continuous):
        super(ActorNetwork, self).__init__()

        self.is_continuous = is_continuous

        self.layers = [nn.Linear(state_dim, fc1_dim), nn.ReLU()]
        self.layers += [nn.Linear(fc1_dim, fc2_dim), nn.ReLU()]
        self.layers = nn.Sequential(*self.layers)

        if is_continuous:
            self.action_mean = nn.Linear(fc2_dim, action_dim)
            self.action_log_std = nn.Linear(fc2_dim, action_dim)
        else:
            self.action_prob = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state):
        x = self.layers(state)

        if self.is_continuous:
            action_mean = F.tanh(self.action_mean(x))
            action_log_std = self.action_log_std(x)
            action_log_std = T.clamp(action_log_std, float('-inf'), 1)

            return action_mean, action_log_std
        else:
            return F.softmax(self.action_prob(x))

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class REINFORCE:
    """
    :param state_dim: 状态空间维度
    :param action_dim: 动作空间维度
    :param ckpt_dir: 保存模型的路径
    :param lr: Actor 网络的学习率
    :param fc1_dim: Actor 网络第一个全连接层的维度
    :param fc2_dim: Actor 网络第二个全连接层的维度
    :param gamma: 折扣因子
    :param is_continuous: 是否为连续动作空间
    """
    def __init__(self, state_dim, action_dim, ckpt_dir, lr=0.0003, fc1_dim=256,
                 fc2_dim=256, gamma=0.99, is_continuous=False):
        self.gamma = gamma
        self.checkpoint_dir = ckpt_dir

        self.main_net = ['Actor-net']

        self.log_prob_memory = []
        self.reward_memory = []

        self.terminate = False
        self.is_continuous = is_continuous

        self.actor = ActorNetwork(lr=lr, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=fc1_dim, fc2_dim=fc2_dim, is_continuous=is_continuous)

    def remember(self, reward, done):
        self.reward_memory.append(reward)
        self.terminate = True if done else False

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(device)

        # 对连续动作空间进行处理
        if self.is_continuous:
            action_mean, action_log_prob = self.actor(state)
            action_mean = action_mean.squeeze()
            action_log_prob = action_log_prob.squeeze()
            action_std = T.exp(action_log_prob)

            dist = Normal(action_mean, action_std)
            action = dist.sample()  # 这会为每个动作维度独立采样

            log_prob = dist.log_prob(action).sum(dim=-1)

            self.log_prob_memory.append(log_prob)
            action = np.ravel(action.detach().cpu())

        # 对离散动作空间进行处理
        else:
            probabilities = self.actor(state).squeeze()
            dist = Categorical(probabilities)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            self.log_prob_memory.append(log_prob)
            action = action.item()

        return action

    def learn(self):
        if not self.terminate:
            return

        R = []
        next_r = 0

        # 计算每个状态的 R 值
        for reward in self.reward_memory[::-1]:
            next_r = reward + self.gamma * next_r
            R.append(next_r)
        R.reverse()

        R = T.tensor(R).to(device)

        R = (R - R.mean()) / (R.std() + 1e-7)  # 归一化

        # 计算策略梯度损失
        loss = -T.mean(T.stack(self.log_prob_memory) * R)

        # 更新 Actor 网络
        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()

        # 清空缓存
        self.log_prob_memory.clear()
        self.reward_memory.clear()

    def save_models(self, episode):
        self.actor.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "actor_{}.pt").format(episode))
        print('Saving actor network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "actor_{}.pt").format(episode))
        print('Loading actor network successfully!')
