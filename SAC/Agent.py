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


class ReplayBuffer:
    """
    :param state_dim: 状态空间维度
    :param action_dim: 动作空间维度
    :param max_size: 经验回放缓冲区的最大容量
    :param batch_size: 从经验回放缓冲区中采样的批次大小
    :param is_continuous: 指示动作空间是连续的还是离散的
    """
    def __init__(self, state_dim, action_dim, max_size, batch_size, is_continuous):
        self.memory_size = max_size
        self.batch_size = batch_size
        self.memory_count = 0

        self.state_memory = np.zeros((self.memory_size, state_dim))
        self.reward_memory = np.zeros((self.memory_size,))
        self.next_state_memory = np.zeros((self.memory_size, state_dim))
        self.terminal_memory = np.zeros((self.memory_size,), dtype=np.bool)

        if not is_continuous:
            self.action_memory = np.zeros((self.memory_size,))
            self.next_action_memory = np.zeros((self.memory_size,))
        else:
            self.action_memory = np.zeros((self.memory_size, action_dim))
            self.next_action_memory = np.zeros((self.memory_size, action_dim))

    def store_transition(self, state, action, reward, state_, done):
        index = self.memory_count % self.memory_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.memory_count += 1

    def sample_buffer(self):
        mem_len = min(self.memory_size, self.memory_count)

        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.memory_count > self.batch_size


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
    :param action_dim: 动作空间维度
    :param fc1_dim: 第一个全连接层的维度
    :param fc2_dim: 第二个全连接层的维度
    :param is_continuous: 是否为连续动作空间
    """
    def __init__(self, lr, state_dim, action_dim, fc1_dim, fc2_dim, is_continuous):
        super(CriticNetwork, self).__init__()

        if is_continuous:
            self.layers = [nn.Linear(state_dim + action_dim, fc1_dim), nn.ReLU()]
            self.layers += [nn.Linear(fc1_dim, fc2_dim), nn.ReLU()]
            self.layers = nn.Sequential(*self.layers)

            self.Q = nn.Linear(fc2_dim, 1)
        else:
            self.layers = [nn.Linear(state_dim, fc1_dim), nn.ReLU()]
            self.layers += [nn.Linear(fc1_dim, fc2_dim), nn.ReLU()]
            self.layers = nn.Sequential(*self.layers)

            self.Q = nn.Linear(fc2_dim, action_dim)

        self.is_continuous = is_continuous
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state, action):

        if self.is_continuous:
            x = T.cat([state, action], dim=-1)
            x = self.layers(x)
        else:
            x = self.layers(state)

        return self.Q(x)

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class SAC:
    """
    :param state_dim: 环境的状态空间维度
    :param action_dim: 环境的动作空间维度
    :param ckpt_dir: 保存模型检查点的目录路径
    :param actor_lr: Actor网络的学习率
    :param critic_lr: Critic网络的学习率
    :param actor_fc1_dim: Actor网络第一层全连接层的维度
    :param actor_fc2_dim: Actor网络第二层全连接层的维度
    :param critic_fc1_dim: Critic网络第一层全连接层的维度
    :param critic_fc2_dim: Critic网络第二层全连接层的维度
    :param gamma: 折扣因子(discount factor)
    :param tau: 软更新目标网络的系数
    :param is_continuous: 是否为连续动作空间
    :param max_size: 经验回放缓冲区的最大容量
    :param batch_size: 从经验回放缓冲区中采样的批次大小
    :param alpha: 初始的温度参数alpha值
    :param temp_lr: 温度参数alpha的学习率
    :param delay_time: 更新Actor网络的延迟步数
    :param target_entropy: 目标熵值,用于计算alpha损失
    """
    def __init__(self, state_dim, action_dim, ckpt_dir, actor_lr=0.0003, critic_lr=0.0003,
                 actor_fc1_dim=256, actor_fc2_dim=256, critic_fc1_dim=256, critic_fc2_dim=256,
                 gamma=0.99, tau=0.005, is_continuous=False, max_size=1000000,
                 batch_size=256, alpha=0.1, temp_lr=0.0003, delay_time=2, target_entropy=0.6):
        self.gamma = gamma
        self.tau = tau
        self.checkpoint_dir = ckpt_dir

        self.main_net = ['Actor-net', 'Critic-net']
        self.batch_size = batch_size

        self.updating_times = 0
        self.delay_time = delay_time
        self.is_continuous = is_continuous

        self.actor = ActorNetwork(lr=actor_lr, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim, is_continuous=is_continuous)
        self.critic_1 = CriticNetwork(lr=critic_lr, state_dim=state_dim, is_continuous=is_continuous,
                                      action_dim=action_dim, fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.critic_2 = CriticNetwork(lr=critic_lr, state_dim=state_dim, is_continuous=is_continuous,
                                      action_dim=action_dim, fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic_1 = CriticNetwork(lr=critic_lr, state_dim=state_dim, is_continuous=is_continuous,
                                             action_dim=action_dim, fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic_2 = CriticNetwork(lr=critic_lr, state_dim=state_dim, is_continuous=is_continuous,
                                             action_dim=action_dim, fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.alpha = alpha
        self.log_alpha = T.tensor(np.log(alpha), dtype=T.float, requires_grad=True)
        self.log_alpha_optimizer = T.optim.Adam([self.log_alpha], lr=temp_lr)

        self.target_entropy = target_entropy * T.log(T.tensor(action_dim))

        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size, is_continuous=is_continuous)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_params, params in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_params.data.copy_(tau * params + (1 - tau) * target_params)
        for target_params, params in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_params.data.copy_(tau * params + (1 - tau) * target_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, isTrain=True):
        state = T.tensor([observation], dtype=T.float).to(device)
        if self.is_continuous:
            action_mean, action_log_prob = self.actor(state)
            action_mean = action_mean.squeeze()
            action_log_prob = action_log_prob.squeeze()

            action_std = T.exp(action_log_prob)

            dist = Normal(action_mean, action_std)
            action = dist.sample()  # 这会为每个动作维度独立采样

            action = np.ravel(action.detach().cpu())
        else:
            probabilities = self.actor(state).squeeze()
            dist = Categorical(probabilities)
            action = dist.sample()

            action = action.item()

        return action

    def evaluate_action(self, state_tensor, action_tensor):
        if self.is_continuous:
            action_mean, action_log_prob = self.actor(state_tensor)
            action_mean = action_mean.squeeze()
            action_log_prob = action_log_prob.squeeze()

            action_std = T.exp(action_log_prob)
            dist = Normal(action_mean, action_std)

            log_prob = dist.log_prob(action_tensor).sum(dim=-1)
        else:
            probabilities = self.actor(state_tensor).squeeze()
            dist = Categorical(probabilities)

            log_prob = dist.log_prob(action_tensor)

        return log_prob

    def learn(self):
        if not self.memory.ready():
            return

        # 从经验回放缓冲区采样批次数据
        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float).to(device).detach()
        actions_tensor = T.tensor(actions, dtype=T.float).to(device).detach()
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device).detach()
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device).detach()
        terminals_tensor = T.tensor(terminals).to(device).detach()

        # 计算目标Q值(使用目标critic网络和下一状态的样本动作)
        with T.no_grad():
            # 对 next_state 进行选择动作
            if self.is_continuous:
                action_mean, action_log_prob = self.actor(next_states_tensor)
                action_mean = action_mean.squeeze()

                action_log_prob = action_log_prob.squeeze()
                action_std = T.exp(action_log_prob)

                dist = Normal(action_mean, action_std)

                next_actions_tensor = dist.sample()
                next_log_prob = dist.log_prob(next_actions_tensor).sum(dim=-1).squeeze()
            else:
                probabilities = self.actor(next_states_tensor).squeeze()
                dist = Categorical(probabilities)

                next_actions_tensor = dist.sample()
                next_log_prob = T.log(probabilities)  # 如果是离散动作空间, 对所有动作概率取 log

            next_q1 = self.target_critic_1.forward(next_states_tensor, next_actions_tensor)
            next_q2 = self.target_critic_2.forward(next_states_tensor, next_actions_tensor)

            if self.is_continuous:
                next_q1 = next_q1.squeeze()
                next_q2 = next_q2.squeeze()

        # 计算当前Q值(使用critic网络和当前状态 - 动作对)
        q1 = self.critic_1.forward(states_tensor, actions_tensor)
        q2 = self.critic_2.forward(states_tensor, actions_tensor)
        if not self.is_continuous:
            q1 = q1[T.arange(q1.size(0)), actions_tensor.long()]
            q2 = q2[T.arange(q2.size(0)), actions_tensor.long()]
        if self.is_continuous:
            q1 = q1.squeeze()
            q2 = q2.squeeze()

        # 计算 soft q
        soft_q1 = next_q1 - T.exp(self.log_alpha) * next_log_prob
        soft_q2 = next_q2 - T.exp(self.log_alpha) * next_log_prob

        # 如果是离散动作空间, 用动作概率 * soft q
        if not self.is_continuous:
            soft_q1 = T.sum(T.exp(next_log_prob) * soft_q1, dim=1).squeeze()
            soft_q2 = T.sum(T.exp(next_log_prob) * soft_q2, dim=1).squeeze()

        soft_q1[terminals_tensor] = 0.0
        soft_q2[terminals_tensor] = 0.0

        target_q1 = rewards_tensor + self.gamma * soft_q1
        target_q2 = rewards_tensor + self.gamma * soft_q2

        critic_1_loss = F.mse_loss(q1, T.min(target_q1, target_q2).detach())
        critic_2_loss = F.mse_loss(q2, T.min(target_q1, target_q2).detach())
        critic_loss = critic_1_loss + critic_2_loss

        # 更新critic网络
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.updating_times += 1
        if self.updating_times % self.delay_time != 0:
            return

        if self.is_continuous:
            action_mean, action_log_prob = self.actor(states_tensor)
            action_mean = action_mean.squeeze()
            action_log_prob = action_log_prob.squeeze()

            action_std = T.exp(action_log_prob)

            dist = Normal(action_mean, action_std)
            actions_tensor = dist.rsample()  # 这会为每个动作维度独立采样

            log_prob = dist.log_prob(actions_tensor).sum(dim=-1).squeeze()
        else:
            probabilities = self.actor(states_tensor).squeeze()
            dist = Categorical(probabilities)
            actions_tensor = dist.sample()

            log_prob = T.log(probabilities)  # 如果是离散动作空间, 对所有动作概率取 log

        q1 = self.critic_1.forward(states_tensor, actions_tensor)
        q2 = self.critic_2.forward(states_tensor, actions_tensor)

        if self.is_continuous:
            q1 = q1.squeeze()
            q2 = q2.squeeze()

        actor_loss = T.exp(self.log_alpha).detach() * log_prob - T.min(q1, q2)

        # 如果是离散动作空间, 用动作概率 * actor_loss
        if not self.is_continuous:
            actor_loss = T.sum(T.exp(log_prob) * actor_loss, dim=1).squeeze()

        # 更新 actor
        self.actor.optimizer.zero_grad()
        actor_loss.mean().backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        # 更新 temperature
        temp_loss = T.exp(self.log_alpha) * (-log_prob.detach() - self.target_entropy)
        if not self.is_continuous:
            temp_loss = T.sum(T.exp(log_prob).detach() * temp_loss, dim=1).squeeze()

        self.log_alpha_optimizer.zero_grad()
        temp_loss.mean().backward()
        self.log_alpha_optimizer.step()

    def save_models(self, episode):
        self.actor.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                "actor_{}.pt").format(episode))
        print('Saving actor network successfully!')
        self.critic_1.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[1],
                                                   "critic_1_{}.pt").format(episode))
        print('Saving critic network successfully!')
        self.critic_2.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[1],
                                                   "critic_2_{}.pt").format(episode))
        print('Saving critic network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                "actor_{}.pt").format(episode))
        print('Loading actor network successfully!')
        self.critic_1.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[1],
                                                   "critic_1_{}.pt").format(episode))
        print('Loading critic network successfully!')
        self.critic_2.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[1],
                                                   "critic_2_{}.pt").format(episode))
        print('Loading critic network successfully!')
