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


class SARSA:
    """
        :param state_dim: 状态空间维度
        :param action_dim: 动作空间维度
        :param ckpt_dir: 保存模型的路径
        :param lr: Q 网络的学习率
        :param fc1_dim: Q 网络第一个全连接层的维度
        :param fc2_dim: Q 网络第二个全连接层的维度
        :param gamma: 折扣因子
        :param epsilon: ε-greedy 策略的初始 ε 值
        :param eps_end: ε-greedy 策略的最小 ε 值
        :param eps_dec: ε 值的衰减率
    """
    def __init__(self, state_dim, action_dim, ckpt_dir, lr=0.0003, fc1_dim=256,
                 fc2_dim=256, epsilon=1.0, eps_end=0.05, eps_dec=5e-4, gamma=0.99):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(action_dim)]

        self.checkpoint_dir = ckpt_dir

        self.main_net = ['Q-net']

        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.next_action = None
        self.terminate = None

        self.q_eval = DeepQNetwork(lr=lr, state_dim=state_dim, action_dim=action_dim,
                                   fc1_dim=fc1_dim, fc2_dim=fc2_dim)

    def remember(self, state, action, reward, state_, action_, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = state_
        self.next_action = action_
        self.terminate = done

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, isTrain=True):
        state = T.tensor([observation], dtype=T.float).to(device)
        q_vals = self.q_eval.forward(state)
        action = T.argmax(q_vals).item()

        if (np.random.random() < self.epsilon) and isTrain:
            # 在训练模式下, 以 ε 的概率选择随机动作
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        states_tensor = T.tensor([self.state], dtype=T.float).to(device)
        action_tensor = T.tensor([self.action], dtype=T.long).to(device)
        reward_tensor = T.tensor([self.reward], dtype=T.float).to(device)
        next_states_tensor = T.tensor([self.next_state], dtype=T.float).to(device)
        next_action_tensor = T.tensor([self.next_action], dtype=T.long).to(device)
        terminate_tensor = T.tensor([self.terminate], dtype=T.bool).to(device)

        with T.no_grad():
            # 计算下一个状态-动作对的 Q 值
            q_ = self.q_eval.forward(next_states_tensor)
            q_[terminate_tensor] = 0.0
            target = reward_tensor + self.gamma * q_[T.arange(q_.size(0)), next_action_tensor]

        # 计算当前状态-动作对的 Q 值
        q = self.q_eval.forward(states_tensor)[T.arange(q_.size(0)), action_tensor]

        # 计算 Q 网络的损失并更新网络
        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        # 衰减 ε 值
        self.decrement_epsilon()

    def save_models(self, episode):
        self.q_eval.save_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "Q_eval_{}.pt").format(episode))
        print('Saving actor network successfully!')

    def load_models(self, episode):
        self.q_eval.load_checkpoint(os.path.join(self.checkpoint_dir, self.main_net[0],
                                                 "Q_eval_{}.pt").format(episode))
        print('Loading actor network successfully!')
