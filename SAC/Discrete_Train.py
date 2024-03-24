import os.path
import gym
import Basic
from Agent import SAC


env_name = 'LunarLander-v2'
env = gym.make(env_name)

# 确保实验的可重复性和结果的一致性
env.seed(42)

# 用于保存模型的文件夹
ckpt_dir = os.path.join(os.curdir, env_name)

agent = SAC(action_dim=env.action_space.n, delay_time=3,
            state_dim=env.observation_space.shape[0], ckpt_dir=ckpt_dir)

Basic.train(env, agent, ckpt_dir, max_episodes=1000)
