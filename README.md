# 基于PyTorch的强化学习算法集合

集合多种基于PyTorch实现的强化学习算法。支持的算法包括但不限于PPO, SAC, DDQN, Dueling DQN, DDPG, TD3, REINFORCE, SARSA, D3QN, A2C, 和DQN。每个算法都配有在LunarLander-v2和LunarLanderContinuous-v2环境下预训练的模型。

## 目录结构

本项目由11个算法文件夹组成，每个文件夹包含以下内容：

- **Agent.py** - 算法实现。
- **Basic.py** - 环境设置。
- **Continuous_Train.py** - 连续动作空间的训练脚本。
- **Discrete_Train.py** - 离散动作空间的训练脚本。
- **Test.py** - 测试脚本。
- **LunarLander-v2/** - 存放在LunarLander-v2环境下预训练的模型。
- **LunarLanderContinuous-v2/** - 存放在LunarLanderContinuous-v2环境下预训练的模型。
