import os.path
import gym
import Basic
from Agent import PPO


env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)

# 确保实验的可重复性和结果的一致性
env.seed(42)

# 用于保存模型的文件夹
ckpt_dir = os.path.join(os.curdir, env_name + "-3000")

agent = PPO(action_dim=env.action_space.shape[0],
            state_dim=env.observation_space.shape[0], ckpt_dir=ckpt_dir,
            is_continuous=True, actor_fc1_dim=128, actor_fc2_dim=128, critic_fc1_dim=128, critic_fc2_dim=128,
            lambda_=0.9, learning_frequency=2000)

Basic.train(env, agent, ckpt_dir, max_episodes=3000)
