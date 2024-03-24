import os.path
import gym
import Basic
from Agent import REINFORCE


env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)

# 用于保存模型的文件夹
ckpt_dir = os.path.join(os.curdir, env_name + "-5000")
episode = 3000  # 读取第 3000 轮训练的模型

agent = REINFORCE(action_dim=env.action_space.shape[0],
                  state_dim=env.observation_space.shape[0],
                  ckpt_dir=ckpt_dir, is_continuous=True)

agent.load_models(episode)
Basic.test(env, agent, max_episodes=10, render=True)
