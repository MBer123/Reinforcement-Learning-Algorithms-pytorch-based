import os.path
import gym
import Basic
from Agent import TD3


env_name = 'LunarLanderContinuous-v2'
env = gym.make(env_name)

# 确保实验的可重复性和结果的一致性
env.seed(42)

# 用于保存模型的文件夹
ckpt_dir = os.path.join(os.curdir, env_name)

agent = TD3(action_dim=env.action_space.shape[0],
            state_dim=env.observation_space.shape[0],
            ckpt_dir=ckpt_dir, actor_fc1_dim=128,
            actor_fc2_dim=128, critic_fc1_dim=128, critic_fc2_dim=128)

Basic.train(env, agent, ckpt_dir, max_episodes=1000)
