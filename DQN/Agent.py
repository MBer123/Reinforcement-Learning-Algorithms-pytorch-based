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
set_seed(0)


class ReplayBuffer:
    """
    :param state_dim: 状态空间维度
    :param max_size: 经验回放缓冲区的最大容量
    :param batch_size: 从经验回放缓冲区中采样的批次大小
    """
    def __init__(self, state_dim, max_size, batch_size):
        self.memory_size = max_size
        self.batch_size = batch_size
        self.memory_count = 0

        self.state_memory = np.zeros((self.memory_size, state_dim))
        self.action_memory = np.zeros((self.memory_size,))
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


class DeepQNetwork(nn.Module):
    """
    :param lr: 学习率
    :param state_dim: 状态空间维度
    :param action_dim: 动作空间维度
    :param fc1_dim: 第一个全连接层的维度
    :param fc2_dim: 第二个全连接层的维度
    """
    def __init__(self, lr, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DeepQNetwork, self).__init__()

        self.layers = [nn.Linear(state_dim, fc1_dim), nn.ReLU()]
        self.layers += [nn.Linear(fc1_dim, fc2_dim), nn.ReLU()]
        self.layers = nn.Sequential(*self.layers)
        self.Q = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, state):
        x = self.layers(state)

        return self.Q(x)

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class DQN:
    """
    :param state_dim: 环境的状态空间维度
    :param action_dim: 环境的动作空间维度
    :param ckpt_dir: 保存模型检查点的目录路径
    :param lr: 学习率(learning rate)
    :param fc1_dim: 第一个全连接层的维度
    :param fc2_dim: 第二个全连接层的维度
    :param gamma: 折扣因子(discount factor)
    :param tau: 软更新目标网络的系数
    :param epsilon: epsilon-greedy策略的初始epsilon值
    :param eps_end: epsilon-greedy策略的最小epsilon值
    :param eps_dec: epsilon递减的步长
    :param max_size: 经验回放缓冲区的最大容量
    :param batch_size: 从经验回放缓冲区中采样的批次大小
    """
    def __init__(self, state_dim, action_dim, ckpt_dir, lr=0.0003, fc1_dim=256, fc2_dim=256,
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.05, eps_dec=5e-4,
                 max_size=1000000, batch_size=256):
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 软更新系数
        self.epsilon = epsilon  # epsilon-greedy策略的初始epsilon值
        self.eps_end = eps_end  # epsilon-greedy策略的最小epsilon值
        self.eps_dec = eps_dec  # epsilon递减的步长
        self.batch_size = batch_size  # 从经验回放缓冲区中采样的批次大小
        self.checkpoint_dir = ckpt_dir  # 保存模型检查点的目录路径
        self.action_space = [i for i in range(action_dim)]  # 动作空间

        self.main_net = ["Q_net"]  # 主网络的名称

        # 初始化评估网络和目标网络
        self.q_eval = DeepQNetwork(lr=lr, state_dim=state_dim, action_dim=action_dim,
                                   fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DeepQNetwork(lr=lr, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=fc1_dim, fc2_dim=fc2_dim)

        self.memory = ReplayBuffer(state_dim=state_dim, max_size=max_size, batch_size=batch_size)  # 初始化经验回放缓冲区

        self.update_network_parameters(tau=1.0)  # 初始化时将目标网络完全复制为评估网络

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # 将评估网络的参数复制到目标网络
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_end else self.eps_end

    def choose_action(self, observation, isTrain=True):
        state = T.tensor([observation], dtype=T.float).to(device)
        q_vals = self.q_eval.forward(state)  # 使用评估网络获取当前状态下各个动作的 Q 值
        action = T.argmax(q_vals).item()  # 选择 Q 值最大的动作

        # 如果处于训练模式,则使用epsilon-greedy策略选择动作
        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if not self.memory.ready():  # 如果经验回放缓冲区中的数据不够,则返回
            return

        # 从经验回放缓冲区中采样批次数据
        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()

        # 将批次数据转换为张量
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.long).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            # 使用目标网络计算下一状态的 Q 值
            q_ = self.q_target.forward(next_states_tensor)
            max_actions = T.argmax(q_, dim=-1)  # 选择 Q 值最大的动作
            q_[terminals_tensor] = 0.0  # 如果是终止状态,Q 值设为 0
            target = rewards_tensor + self.gamma * q_[T.arange(q_.size(0)), max_actions]  # 计算目标Q值

        # 计算当前状态下采取动作的Q值
        q = self.q_eval.forward(states_tensor)[T.arange(q_.size(0)), actions_tensor]

        # 计算损失
        loss = F.mse_loss(q, target.detach())

        # 优化评估网络
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        # 软更新目标网络
        self.update_network_parameters()

        # 更新epsilon值
        self.decrement_epsilon()

    def save_models(self, episode):
        self.q_eval.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "Q_eval_{}.pt").format(episode))
        print('Saving Q_eval network successfully!')

    def load_models(self, episode):
        self.q_eval.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "Q_eval_{}.pt").format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                   "Q_eval_{}.pt").format(episode))
        print('Loading Q_eval network successfully!')
