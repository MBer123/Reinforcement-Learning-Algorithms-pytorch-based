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

        self.layers = [nn.Linear(state_dim, fc1_dim), nn.Tanh()]
        self.layers += [nn.Linear(fc1_dim, fc2_dim), nn.Tanh()]
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


class CriticNetwork(nn.Module):
    """
    :param lr: 学习率
    :param state_dim: 状态空间维度
    :param fc1_dim: 第一个全连接层的维度
    :param fc2_dim: 第二个全连接层的维度
    """
    def __init__(self, lr, state_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.layers = [nn.Linear(state_dim, fc1_dim), nn.ReLU()]
        self.layers += [nn.Linear(fc1_dim, fc2_dim), nn.ReLU()]
        self.layers = nn.Sequential(*self.layers)

        self.V = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state):
        x = self.layers(state)

        return self.V(x)

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class A2C:
    """
    :param state_dim: 环境状态空间的维度
    :param action_dim: 环境动作空间的维度
    :param ckpt_dir: 保存模型权重文件的路径
    :param gamma: 折扣因子
    :param is_continuous: 指示动作空间是连续的还是离散的
    :param actor_lr: Actor网络的学习率
    :param critic_lr: Critic网络的学习率
    :param actor_fc1_dim: Actor网络第一个全连接层的维度
    :param actor_fc2_dim: Actor网络第二个全连接层的维度
    :param critic_fc1_dim: Critic网络第一个全连接层的维度
    :param critic_fc2_dim: Critic网络第二个全连接层的维度
    :param learning_frequency: 多少步后进行一次学习
    :param tau: 软更新目标网络的系数
    :param delay_time: 策略更新延迟的时间步数
    """
    def __init__(self, state_dim, action_dim, ckpt_dir, actor_lr=0.0003, critic_lr=0.0003,
                 actor_fc1_dim=256, actor_fc2_dim=256, critic_fc1_dim=256, critic_fc2_dim=256,
                 gamma=0.99, tau=0.005, is_continuous=False, learning_frequency=1000, delay_time=2):
        self.gamma = gamma
        self.lambda_ = 0.95
        self.tau = tau
        self.checkpoint_dir = ckpt_dir
        self.delay_time = delay_time * learning_frequency

        self.main_net = ['Actor-net', 'Critic-net']
        self.state_memory = []
        self.reward_memory = []
        self.log_prob_memory = []
        self.dist_entropy_memory = []
        self.next_state_memory = []
        self.terminate_memory = []

        self.count = 0
        self.is_continuous = is_continuous
        self.learning_frequency = learning_frequency

        self.actor = ActorNetwork(lr=actor_lr, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim, is_continuous=is_continuous)
        self.critic = CriticNetwork(lr=critic_lr, state_dim=state_dim,
                                    fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

    def remember(self, state, reward, state_, done):
        self.state_memory.append(state)
        self.reward_memory.append(reward)
        self.next_state_memory.append(state_)
        self.terminate_memory.append(done)
        self.count += 1

    def choose_action(self, observation, isTrain=True):
        state = T.tensor([observation], dtype=T.float).to(device)

        # 如果是连续动作空间
        if self.is_continuous:
            action_mean, action_log_prob = self.actor(state)
            action_mean = action_mean.squeeze()
            action_log_prob = action_log_prob.squeeze()

            action_std = T.exp(action_log_prob)

            dist = Normal(action_mean, action_std)
            action = dist.sample()  # 这会为每个动作维度独立采样

            log_prob = dist.log_prob(action).sum(dim=-1).squeeze()
            self.log_prob_memory.append(log_prob)

            dist_entropy = dist.entropy().sum(dim=-1).squeeze()
            self.dist_entropy_memory.append(dist_entropy)

            action = np.ravel(action.detach().cpu())

        # 如果是离散动作空间
        else:
            probabilities = self.actor(state).squeeze()
            dist = Categorical(probabilities)
            action = dist.sample()

            log_prob = dist.log_prob(action).squeeze()
            self.log_prob_memory.append(log_prob)

            dist_entropy = dist.entropy().squeeze()
            self.dist_entropy_memory.append(dist_entropy)

            action = action.item()

        return action

    def learn(self):
        if self.count % self.learning_frequency != 0:
            return

        states_tensor = T.tensor(self.state_memory, dtype=T.float).to(device)

        R = []
        next_r = 0

        # 计算每个时间步的折现累计回报
        for reward, done in zip(self.reward_memory[::-1], self.terminate_memory[::-1]):
            if done:
                next_r = 0
            next_r = reward + self.gamma * next_r
            R.append(next_r)
        R.reverse()

        v = self.critic(states_tensor).view(-1)
        advantage = T.tensor(R).to(device) - v  # 计算优势函数

        # 更新Critic网络
        critic_loss = T.mean(advantage.pow(2))
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        if self.count % self.delay_time != 0:
            self.state_memory.clear()
            self.log_prob_memory.clear()
            self.reward_memory.clear()
            self.next_state_memory.clear()
            self.dist_entropy_memory.clear()
            self.terminate_memory.clear()
            return

        # 更新Actor网络
        actor_loss = -T.mean(
            T.stack(self.log_prob_memory) * advantage.detach() + 0.01 * T.stack(self.dist_entropy_memory))

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.state_memory.clear()
        self.log_prob_memory.clear()
        self.reward_memory.clear()
        self.next_state_memory.clear()
        self.dist_entropy_memory.clear()
        self.terminate_memory.clear()

    def save_models(self, episode):
        self.actor.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "actor_{}.pt").format(episode))
        print('Saving actor network successfully!')
        self.critic.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[1],
                                                 "critic_{}.pt").format(episode))
        print('Saving critic network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "actor_{}.pt").format(episode))
        print('Loading actor network successfully!')
        self.critic.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[1],
                                                 "critic_{}.pt").format(episode))
        print('Loading critic network successfully!')
