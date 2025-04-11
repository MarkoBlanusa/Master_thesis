import json

# Replace with your actual file path
file_path = r"C:\Users\marko\ray_results\PPO_2024-08-30_15-13-25\PPO_trading_env_dynamic_numpy-v0_a3f01_00000_0_2024-08-30_15-13-25/result.json"

with open(file_path) as f:
    for line in f:
        data = json.loads(line)
        print(json.dumps(data, indent=4))  # Nicely formatted output
