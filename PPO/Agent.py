import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
from torch.distributions import Categorical, Normal, MultivariateNormal


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
        self.layers = [nn.Linear(state_dim, fc1_dim), nn.Tanh()]
        self.layers += [nn.Linear(fc1_dim, fc2_dim), nn.Tanh()]
        self.layers = nn.Sequential(*self.layers)

        self.V = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.to(device)

    def forward(self, state):
        x = self.layers(state)

        return self.V(x)

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class PPO:
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
    :param K_epochs: 每次学习过程中的epochs数量
    :param epsilon_clip: 裁剪参数,控制新旧策略偏离程度
    :param learning_frequency: 多少步后进行一次学习
    """
    def __init__(self, state_dim, action_dim, ckpt_dir, actor_lr=0.0003, critic_lr=0.0003,
                 actor_fc1_dim=256, actor_fc2_dim=256, critic_fc1_dim=256, critic_fc2_dim=256, ent_coef=0.01,
                 gamma=0.99, K_epochs=80, epsilon_clip=0.1, is_continuous=False, learning_frequency=1000,
                 lambda_=0.95):
        self.checkpoint_dir = ckpt_dir

        self.main_net = ['Actor-net', 'Critic-net']
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon_clip = epsilon_clip
        self.K_epochs = K_epochs
        self.ent_coef = ent_coef

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.next_state_memory = []
        self.log_prob_memory = []
        self.terminate_memory = []

        self.count = 0
        self.is_continuous = is_continuous
        self.learning_frequency = learning_frequency

        # 初始化Actor网络和Critic网络
        self.actor = ActorNetwork(lr=actor_lr, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim, is_continuous=is_continuous)
        self.critic = CriticNetwork(lr=critic_lr, state_dim=state_dim,
                                    fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

    def remember(self, state, action, reward, state_, done):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.next_state_memory.append(state_)
        self.terminate_memory.append(done)
        self.count += 1

    def choose_action(self, observation, isTrain=True):
        state = T.tensor([observation], dtype=T.float).to(device)
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
        else:
            probabilities = self.actor(state).squeeze()
            dist = Categorical(probabilities)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            self.log_prob_memory.append(log_prob)

            action = action.item()

        return action

    def evaluate_action(self, state_tensor, action_tensor):
        if self.is_continuous:
            action_mean, action_log_prob = self.actor(state_tensor)
            action_mean = action_mean.squeeze()
            action_log_prob = action_log_prob.squeeze()

            action_std = T.exp(action_log_prob)
            dist = Normal(action_mean, action_std)

            dist_entropy = dist.entropy().sum(dim=-1)
            log_prob = dist.log_prob(action_tensor).sum(dim=-1)
        else:
            probabilities = self.actor(state_tensor).squeeze()
            dist = Categorical(probabilities)

            dist_entropy = dist.entropy()
            log_prob = dist.log_prob(action_tensor)

        return log_prob, dist_entropy

    def learn(self):
        if self.count < self.learning_frequency:
            return

        # 转换为张量
        state_array = np.array(self.state_memory)
        state_tensor = T.tensor(state_array, dtype=T.float).to(device)

        action_tensor = T.tensor(self.action_memory, dtype=T.float).to(device)
        reward_tensor = T.tensor(self.reward_memory, dtype=T.float).to(device)

        with T.no_grad():
            log_prob_tensor = T.tensor(self.log_prob_memory, dtype=T.float).squeeze().to(device)
            state_value = self.critic(state_tensor).squeeze().to(device)
            next_state_value = self.critic(next_state_tensor).squeeze().to(device)
            terminate_tensor = T.tensor(self.terminate_memory, dtype=T.float).squeeze().to(device)

            delta = reward_tensor + self.gamma * next_state_value * (1 - terminate_tensor) - state_value

            gae = 0
            advantage = []
            gamma_lambda = self.gamma * self.lambda_
            for delta in reversed(delta.detach().cpu().numpy()):
                gae = delta + gamma_lambda * gae
                advantage.insert(0, gae)

            advantage = T.tensor(advantage, dtype=T.float32).to(device)
            R = advantage + state_value

        for _ in range(self.K_epochs):
            # 评估新的对数概率和熵
            new_log_prob_tensor, dist_entropy = self.evaluate_action(state_tensor, action_tensor)
            new_log_prob_tensor = new_log_prob_tensor.squeeze()
            dist_entropy = dist_entropy.squeeze()

            # 计算新的状态值
            new_state_value = self.critic(state_tensor).squeeze()

            # 计算重要性采样比率
            ratios = T.exp(new_log_prob_tensor - log_prob_tensor.detach())

            # 计算代理损失和值损失
            surrogate_1 = ratios * advantage
            surrogate_2 = T.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage

            actor_loss = -T.min(surrogate_1, surrogate_2)
            critic_loss = F.mse_loss(new_state_value, R)

            # 总损失 = actor 损失 + critic 损失 - 熵损失
            loss = T.mean(actor_loss + 0.5 * critic_loss - self.ent_coef * dist_entropy)

            self.critic.optimizer.zero_grad()
            self.actor.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()
            self.actor.optimizer.step()

        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()
        self.next_state_memory.clear()
        self.log_prob_memory.clear()
        self.terminate_memory.clear()
        self.count = 0

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
