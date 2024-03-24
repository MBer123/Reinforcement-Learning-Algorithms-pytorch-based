import numpy as np
import os
import pandas as pd


def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        sub_path = os.path.join(path, sub_dir)
        if os.path.exists(sub_path):
            print(sub_path + 'is already exist!')
        else:
            os.makedirs(sub_path, exist_ok=True)
            print(sub_path + 'create successfully!')


def train(env, agent, ckpt_dir, max_episodes=500):
    # 创建文件夹用于保存模型
    create_directory(ckpt_dir, sub_dirs=agent.main_net)

    total_rewards, avg_rewards = [], []

    for episode in range(max_episodes):
        total_reward = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation, isTrain=True)
            observation_, reward, done, info = env.step(action)

            agent.remember(observation, reward, observation_, done)
            agent.learn(,

            total_reward += reward
            observation = observation_

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        print('EP:{} Reward:{} Avg_reward:{}'.
              format(episode + 1, total_reward, avg_reward))

        if (episode + 1) % 50 == 0:
            agent.save_models(episode + 1)

    df = pd.DataFrame(total_rewards, columns=['Total Rewards'])
    df.to_csv(os.path.join(ckpt_dir, "returns.csv"), index=False)


def test(env, agent, max_episodes=10, render=True):
    total_rewards, avg_rewards = [], []

    for episode in range(max_episodes):
        total_reward = 0
        done = False
        observation = env.reset()
        while not done:
            if render:
                env.render()

            action = agent.choose_action(observation, isTrain=False)
            observation_, reward, done, info = env.step(action)

            total_reward += reward
            observation = observation_

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        print('EP:{} Reward:{} Avg_reward:{} '.
              format(episode + 1, total_reward, avg_reward))
