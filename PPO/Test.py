import os.path
import gym
import Basic
from Agent import PPO


env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)

# 用于保存模型的文件夹
ckpt_dir = os.path.join(os.curdir, env_name + "-3000")
episode = 1750  # 读取第 900 轮训练的模型

agent = PPO(action_dim=env.action_space.shape[0],
            state_dim=env.observation_space.shape[0], ckpt_dir=ckpt_dir,
            is_continuous=True, actor_fc1_dim=128, actor_fc2_dim=128, critic_fc1_dim=128, critic_fc2_dim=128,
            lambda_=0.9, learning_frequency=2000)

agent.load_models(episode)
Basic.test(env, agent, max_episodes=10, render=True)
