import pandas as pd
import matplotlib.pyplot as plt

# Load the two CSV files
# file_path_1 = r"C:\Users\marko\ray_results\PPO_2025-03-03_21-45-03\PPO_trade_env_ray_portfolio_61c08_00000_0_2025-03-03_21-45-03\progress.csv"
# file_path_2 = r"C:\Users\marko\ray_results\PPO_2025-03-04_02-18-13\PPO_trade_env_ray_portfolio_8ae78_00000_0_2025-03-04_02-18-13\progress.csv"
# file_path_1 = r"C:\Users\marko\ray_results\PPO_2025-03-08_15-17-56\PPO_trade_env_ray_portfolio_2182b_00000_0_2025-03-08_15-17-56\progress.csv"
# file_path_2 = r"D:\Master_thesis\DRL_method_results\PPO_2025-03-09_20-13-13_iterative_constrained\PPO_trade_env_ray_portfolio_8c0de_00000_0_2025-03-09_20-13-13\progress.csv"
# file_path_3 = r"D:\Master_thesis\DRL_method_results\PPO_2025-03-11_14-20-38_iterative_constrained_2\PPO_trade_env_ray_portfolio_9f80a_00000_0_2025-03-11_14-20-38\progress.csv"
# file_path_4 = r"C:\Users\marko\ray_results\PPO_2025-03-12_03-35-25\PPO_trade_env_ray_portfolio_a6ffb_00000_0_2025-03-12_03-35-25\progress.csv"
file_path_7 = r"C:\Users\marko\ray_results\PPO_2025-03-16_04-57-44_min_variance_no_rolling\PPO_trade_env_ray_portfolio_d0e6a_00000_0_2025-03-16_04-57-44\progress.csv"
file_path_8 = r"C:\Users\marko\ray_results\PPO_2025-03-17_00-17-08_Min_variance_net_exp_1_no_rolling\PPO_trade_env_ray_portfolio_c8010_00000_0_2025-03-17_00-17-08\progress.csv"
# file_path_9 = r"D:\ray_results\smaller_lstm_penalties_smaller_9_period_2\PPO_trading_env_dynamic_numpy-v0_443f7_00000_0_2024-09-08_01-59-49\progress.csv"
# file_path_10 = r"D:\ray_results\smaller_lstm_penalties_smaller_10_period_2\PPO_trading_env_dynamic_numpy-v0_eb197_00000_0_2024-09-08_11-01-22\progress.csv"
# file_path_11 = r"D:\ray_results\smaller_lstm_penalties_smaller_11_period_3\PPO_trading_env_dynamic_numpy-v0_c8f99_00000_0_2024-09-08_14-20-50\progress.csv"
# file_path_12 = r"D:\ray_results\smaller_lstm_penalties_smaller_12_period_3\PPO_trading_env_dynamic_numpy-v0_723c1_00000_0_2024-09-08_23-29-36\progress.csv"
# file_path_13 = r"D:\ray_results\smaller_lstm_penalties_smaller_13_period_3\PPO_trading_env_dynamic_numpy-v0_5057d_00000_0_2024-09-09_10-48-41\progress.csv"
# file_path_14 = r"D:\ray_results\smaller_lstm_penalties_smaller_14_period_3\PPO_trading_env_dynamic_numpy-v0_0e6e6_00000_0_2024-09-09_22-06-53\progress.csv"
# file_path_15 = r"D:\ray_results\smaller_lstm_penalties_smaller_15_period_3\PPO_trading_env_dynamic_numpy-v0_ba672_00000_0_2024-09-09_23-44-45\progress.csv"
# file_path_16 = r"D:\ray_results\smaller_lstm_penalties_smaller_16_period_4\PPO_trading_env_dynamic_numpy-v0_c51bd_00000_0_2024-09-10_03-34-07\progress.csv"
# file_path_17 = r"D:\ray_results\smaller_lstm_penalties_smaller_17_period_4\PPO_trading_env_dynamic_numpy-v0_870bc_00000_0_2024-09-10_13-19-21\progress.csv"
# file_path_18 = r"D:\ray_results\smaller_lstm_penalties_smaller_18_period_4\PPO_trading_env_dynamic_numpy-v0_bbd86_00000_0_2024-09-10_23-22-08\progress.csv"
# file_path_19 = r"D:\ray_results\smaller_lstm_penalties_smaller_19_period_4\PPO_trading_env_dynamic_numpy-v0_a6fac_00000_0_2024-09-11_09-30-00\progress.csv"
# file_path_20 = r"C:\Users\marko\ray_results\smaller_lstm_penalties_smaller_20_period_4\PPO_trading_env_dynamic_numpy-v0_d1ef5_00000_0_2024-09-11_21-48-30\progress.csv"
# file_path_21 = r"D:\ray_results\smaller_lstm_penalties_smaller_21_period_4\PPO_trading_env_dynamic_numpy-v0_8ab9f_00000_0_2024-09-12_12-34-08\progress.csv"
# file_path_22 = r"C:\Users\marko\ray_results\PPO_2024-09-12_16-11-30\PPO_trading_env_dynamic_numpy-v0_e82ac_00000_0_2024-09-12_16-11-30\progress.csv"


# data_1 = pd.read_csv(file_path_1)
# data_2 = pd.read_csv(file_path_2)
# data_3 = pd.read_csv(file_path_3)
# data_4 = pd.read_csv(file_path_4)
# data_5 = pd.read_csv(file_path_5)
# data_6 = pd.read_csv(file_path_6)
data_7 = pd.read_csv(file_path_7)
data_8 = pd.read_csv(file_path_8)
# data_9 = pd.read_csv(file_path_9)
# data_10 = pd.read_csv(file_path_10)
# data_11 = pd.read_csv(file_path_11)
# data_12 = pd.read_csv(file_path_12)
# data_13 = pd.read_csv(file_path_13)
# data_14 = pd.read_csv(file_path_14)
# data_15 = pd.read_csv(file_path_15)
# data_16 = pd.read_csv(file_path_16)
# data_17 = pd.read_csv(file_path_17)
# data_18 = pd.read_csv(file_path_18)
# data_19 = pd.read_csv(file_path_19)
# data_20 = pd.read_csv(file_path_20)
# data_21 = pd.read_csv(file_path_21)
# data_22 = pd.read_csv(file_path_22)


# Combine the data by appending data_2 to data_1
combined_data = data_8
# combined_data = pd.concat([data_7, data_8], ignore_index=True)
# combined_data = pd.concat([combined_data, data_3], ignore_index=True)
# combined_data = pd.concat([combined_data, data_4], ignore_index=True)
# combined_data = pd.concat([combined_data, data_5], ignore_index=True)
# combined_data = pd.concat([combined_data, data_6], ignore_index=True)
# combined_data = pd.concat([combined_data, data_7], ignore_index=True)
# combined_data = pd.concat([combined_data, data_8], ignore_index=True)
# combined_data = pd.concat([combined_data, data_9], ignore_index=True)
# combined_data = pd.concat([combined_data, data_10], ignore_index=True)
# combined_data = pd.concat([combined_data, data_11], ignore_index=True)
# combined_data = pd.concat([combined_data, data_12], ignore_index=True)
# combined_data = pd.concat([combined_data, data_13], ignore_index=True)
# combined_data = pd.concat([combined_data, data_14], ignore_index=True)
# combined_data = pd.concat([combined_data, data_15], ignore_index=True)
# combined_data = pd.concat([combined_data, data_16], ignore_index=True)
# combined_data = pd.concat([combined_data, data_17], ignore_index=True)
# combined_data = pd.concat([combined_data, data_18], ignore_index=True)
# combined_data = pd.concat([combined_data, data_19], ignore_index=True)
# combined_data = pd.concat([combined_data, data_20], ignore_index=True)
# combined_data = pd.concat([combined_data, data_21], ignore_index=True)
# combined_data = pd.concat([combined_data, data_22], ignore_index=True)


# Save the combined data to a new CSV file
combined_file_path = "combined_data.csv"
combined_data.to_csv(combined_file_path, index=False)

# Extract the relevant columns for plotting
data_plot = combined_data[["episode_reward_mean", "episodes_this_iter"]]

# Find the 10 highest rewards and their corresponding iterations
top_10_rewards = combined_data.nlargest(25, "episode_reward_mean")[
    ["episode_reward_mean", "episodes_this_iter"]
]

# Print the top 10 rewards and iterations
print("Top 10 highest rewards and their corresponding iterations:")
print(top_10_rewards)

# Plotting mean reward per iteration
plt.figure(figsize=(10, 6))
plt.plot(
    data_plot.index + 1,
    data_plot["episode_reward_mean"],
    marker="o",
    markersize=1,
    linewidth=1.0,
)
plt.title("Mean Reward per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Mean Reward")
plt.grid(True)
plt.show()
