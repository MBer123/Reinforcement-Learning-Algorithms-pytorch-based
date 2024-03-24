import pandas as pd
import matplotlib.pyplot as plt
import os


# env_name = "LunarLander-v2"
# algorithms = ["SARSA", "A2C", "D3QN", "DDQN", "DuelingDQN", "PPO", "REINFORCE", "SAC"]
env_name = "LunarLanderContinuous-v2"
algorithms = ['REINFORCE', "DDPG", "TD3", "PPO", "SAC", "A2C"]

plt.figure(figsize=(10, 6))

# 对于每个CSV文件
for algorithm in algorithms:
    # 加载CSV文件
    df = pd.read_csv(os.path.join(algorithm, env_name, "returns.csv"))

    # 计算移动平均值
    rolling_mean = df['Total Rewards'].rolling(window=100, min_periods=1).mean()

    # 绘制每次奖励的曲线，透明度为30%
    plt.plot(df['Total Rewards'], alpha=0.3)

    # 绘制移动平均值的曲线，不设置透明度
    plt.plot(rolling_mean, label=algorithm)

# 添加网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

# 添加标题和标签
plt.title('Agent Performance Comparison')
plt.xlabel('Episodes')
plt.ylabel('Average Return')

# 显示图例
plt.legend()

# 显示图表
plt.show()
