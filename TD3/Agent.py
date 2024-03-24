import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    """
    def __init__(self, state_dim, action_dim, max_size, batch_size):
        self.memory_size = max_size
        self.batch_size = batch_size
        self.memory_count = 0

        self.state_memory = np.zeros((self.memory_size, state_dim))
        self.action_memory = np.zeros((self.memory_size, action_dim))
        self.reward_memory = np.zeros((self.memory_size,))
        self.next_state_memory = np.zeros((self.memory_size, state_dim))
        self.terminal_memory = np.zeros((self.memory_size,), dtype=np.bool)

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
    """
    def __init__(self, lr, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()

        self.layers = [nn.Linear(state_dim, fc1_dim), nn.ReLU()]
        self.layers += [nn.Linear(fc1_dim, fc2_dim), nn.ReLU()]
        self.layers = nn.Sequential(*self.layers)
        self.action = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state):
        x = self.layers(state)
        action = self.action(x)

        return F.tanh(action)

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
    """
    def __init__(self, lr, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.layers = [nn.Linear(state_dim + action_dim, fc1_dim), nn.ReLU()]
        self.layers += [nn.Linear(fc1_dim, fc2_dim), nn.ReLU()]
        self.layers = nn.Sequential(*self.layers)
        self.Q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state, action):
        # 将状态和动作拼接在一起
        x = T.cat([state, action], dim=1)
        x = self.layers(x)

        return self.Q(x)

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class TD3:
    """
    :param state_dim: 状态空间维度
    :param action_dim: 动作空间维度
    :param ckpt_dir: 保存模型的路径
    :param actor_lr: Actor 网络的学习率
    :param critic_lr: Critic 网络的学习率
    :param actor_fc1_dim: Actor 网络第一个全连接层的维度
    :param actor_fc2_dim: Actor 网络第二个全连接层的维度
    :param critic_fc1_dim: Critic 网络第一个全连接层的维度
    :param critic_fc2_dim: Critic 网络第二个全连接层的维度
    :param action_noise: 动作噪声的标准差
    :param policy_noise: 目标策略平滑正则化噪声的标准差
    :param policy_noise_clip: 目标策略平滑正则化噪声的裁剪阈值
    :param delay_time: 策略更新延迟的时间步数
    :param gamma: 折扣因子
    :param tau: 软更新系数
    :param max_size: 经验回放缓冲区的最大容量
    :param batch_size: 从经验回放缓冲区中采样的批次大小
    """
    def __init__(self, state_dim, action_dim, ckpt_dir, actor_lr=0.0003, critic_lr=0.0003, actor_fc1_dim=256,
                 actor_fc2_dim=256, critic_fc1_dim=256, critic_fc2_dim=256, action_noise=0.1, policy_noise=0.2,
                 policy_noise_clip=0.5, delay_time=2, gamma=0.99, tau=0.005, max_size=1000000, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip

        self.delay_time = delay_time
        self.updating_times = 0
        self.checkpoint_dir = ckpt_dir

        self.main_net = ['Actor-net', 'Critic-1-net', 'Critic-2-net']

        self.actor = ActorNetwork(lr=actor_lr, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.target_actor = ActorNetwork(lr=actor_lr, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic_1 = CriticNetwork(lr=critic_lr, state_dim=state_dim, action_dim=action_dim,
                                      fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic_1 = CriticNetwork(lr=critic_lr, state_dim=state_dim, action_dim=action_dim,
                                             fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.critic_2 = CriticNetwork(lr=critic_lr, state_dim=state_dim, action_dim=action_dim,
                                      fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic_2 = CriticNetwork(lr=critic_lr, state_dim=state_dim, action_dim=action_dim,
                                             fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim,
                                   action_dim=action_dim, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # 更新目标 Actor 网络的参数
        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        # 更新目标 Critic 网络 1 的参数
        for critic_params, target_critic_params in zip(self.critic_1.parameters(),
                                                       self.target_critic_1.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)

        # 更新目标 Critic 网络 2 的参数
        for critic_params, target_critic_params in zip(self.critic_2.parameters(),
                                                       self.target_critic_2.parameters()):
            target_critic_params.data.copy_(tau * critic_params + (1 - tau) * target_critic_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, isTrain=True):
        state = T.tensor([observation], dtype=T.float).to(device)
        action = self.actor(state).squeeze()

        if isTrain:
            # 在训练模式下添加探索噪声
            noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=T.float).to(device)

            """将 action 映射到有效范围内"""
            action = T.clamp(action + noise, -1, 1)

        return np.ravel(action.detach().cpu())

    def learn(self):
        if not self.memory.ready():
            return

        self.updating_times += 1
        states, actions, reward, states_, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float).to(device)
        rewards_tensor = T.tensor(reward, dtype=T.float).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            # 计算下一个状态的动作
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            action_noise = T.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                    dtype=T.float).to(device)
            # 对动作噪声进行裁剪
            action_noise = T.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_actions_tensor = T.clamp(next_actions_tensor + action_noise, -1, 1)
            q1_ = self.target_critic_1.forward(next_states_tensor, next_actions_tensor).view(-1)
            q2_ = self.target_critic_2.forward(next_states_tensor, next_actions_tensor).view(-1)
            q1_[terminals_tensor] = 0.0
            q2_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * T.min(q1_, q2_)

        # 计算当前状态的 Q 值
        q1 = self.critic_1.forward(states_tensor, actions_tensor).view(-1)
        q2 = self.critic_2.forward(states_tensor, actions_tensor).view(-1)

        # 计算 Critic 损失并更新 Critic 网络
        critic1_loss = F.mse_loss(q1, target.detach())
        critic2_loss = F.mse_loss(q2, target.detach())
        critic_loss = critic1_loss + critic2_loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.updating_times += 1
        if self.updating_times % self.delay_time != 0:
            return

        # 计算 Actor 损失并更新 Actor 网络
        new_actions_tensor = self.actor.forward(states_tensor)
        q1 = self.critic_1.forward(states_tensor, new_actions_tensor)
        actor_loss = -T.mean(q1)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # 软更新目标网络的参数
        self.update_network_parameters()

    def save_models(self, episode):
        self.actor.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "actor_{}.pt").format(episode))
        print('Saving actor network successfully!')
        self.critic_1.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[1],
                                                   "critic_1_{}.pt").format(episode))
        print('Saving critic 1 network successfully!')
        self.critic_2.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[2],
                                                   "critic_2_{}.pt").format(episode))
        print('Saving critic 2 network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "actor_{}.pt").format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                       "actor_{}.pt").format(episode))
        print('Loading target_actor network successfully!')
        self.critic_1.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[1],
                                                   "critic_1_{}.pt").format(episode))
        print('Loading critic network successfully!')
        self.target_critic_1.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[1],
                                                          "critic_1_{}.pt").format(episode))
        print('Loading target critic network successfully!')
        self.critic_2.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[2],
                                                   "critic_2_{}.pt").format(episode))
        print('Loading critic network successfully!')
        self.target_critic_2.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[2],
                                                          "critic_2_{}.pt").format(episode))
        print('Loading target critic network successfully!')
