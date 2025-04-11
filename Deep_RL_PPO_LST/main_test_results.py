import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime
from typing import Dict
import gc
import logging
import json
import os

from database import Hdf5client
import utils


import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search import basic_variant
from ray.data import Dataset, from_numpy
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.callbacks import DefaultCallbacks


from torch.utils.data import Dataset, DataLoader
import torch


from gymnasium.envs.registration import register
import gymnasium


# initialize torch and neural networks
torch, nn = try_import_torch()


# Data retrieving
def get_timeframe_data(symbol, from_time, to_time, timeframe):
    h5_db = Hdf5client("binance")
    data = h5_db.get_data(symbol, from_time, to_time)
    if timeframe != "1m":
        data = utils.resample_timeframe(data, timeframe)
    return data


# Features engineering
def calculate_indicators(df):
    df["MA5"] = df["close"].rolling(window=5).mean()
    df["MA10"] = df["close"].rolling(window=10).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2

    df["BB_up"] = df["MA10"] + 2 * df["close"].rolling(window=10).std()
    df["BB_low"] = df["MA10"] - 2 * df["close"].rolling(window=10).std()

    df["high-low"] = df["high"] - df["low"]
    df["high-close"] = np.abs(df["high"] - df["close"].shift())
    df["low-close"] = np.abs(df["low"] - df["close"].shift())
    df["TR"] = df[["high-low", "high-close", "low-close"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=14).mean()

    df.drop(["high-low", "high-close", "low-close", "TR"], axis=1, inplace=True)

    return df.dropna()


# Normalize the dataframes
def normalize_dataframes(
    data,
    ohlc_columns=["open", "high", "low", "close"],
    volume_column="volume",
    indicator_columns=["MA5", "MA10", "RSI", "MACD", "BB_up", "BB_low", "ATR"],
    epsilon=0.0001,  # Small constant to avoid zero in normalized data
):
    """
    Normalize the features of financial dataframes.

    :param data: A dictionary of pandas dataframes keyed by timeframe.
    :param ohlc_columns: List of columns to be normalized across all dataframes together.
    :param volume_column: The volume column to be normalized independently for each dataframe.
    :param indicator_columns: List of other indicator columns to normalize independently for each dataframe.
    :param epsilon: Small constant to set the lower bound of the normalized range.
    :return: The dictionary of normalized dataframes and the OHLC scaler used.
    """
    # Initialize the scalers
    ohlc_scaler = MinMaxScaler(
        feature_range=(epsilon, 1)
    )  # Set feature range with epsilon
    volume_scaler = MinMaxScaler(feature_range=(epsilon, 1))

    # Create a new dictionary to store the normalized dataframes
    normalized_data = {}

    # Normalize OHLC data across all timeframes together
    combined_ohlc = pd.concat([df[ohlc_columns] for df in data.values()], axis=0)
    scaled_ohlc = ohlc_scaler.fit_transform(combined_ohlc).astype(np.float32)

    # Distribute the normalized OHLC values back to the original dataframes
    start_idx = 0
    for tf, df in data.items():
        end_idx = start_idx + len(df)
        # Create a copy of the original dataframe to avoid modifying it
        normalized_df = df.copy()
        normalized_df[ohlc_columns] = scaled_ohlc[start_idx:end_idx]
        # Store the normalized dataframe in the new dictionary
        normalized_data[tf] = normalized_df
        start_idx = end_idx

    # Normalize volume independently for each timeframe
    for tf, df in normalized_data.items():
        volume_scaler = MinMaxScaler(
            feature_range=(epsilon, 1)
        )  # Reinitialize scaler for each dataframe
        df[volume_column] = volume_scaler.fit_transform(df[[volume_column]])

    # Normalize other indicators independently for each indicator within each timeframe
    for tf, df in normalized_data.items():
        for col in indicator_columns:
            if col in df.columns:
                scaler = MinMaxScaler(feature_range=(epsilon, 1))
                df[[col]] = scaler.fit_transform(df[[col]])

    return normalized_data, ohlc_scaler


# Add identifiers for the timeframes in order to help the LSTM to make the difference
# def add_timeframe_identifier(data_dict):
#    timeframe_ids = {
#        "15m": 0,
#        "30m": 1,
#        "1h": 2,
#        "1d": 3,
#    }
#    for timeframe, data in data_dict.items():
#        # Assuming `data` is a DataFrame
#        data["timeframe_id"] = timeframe_ids[timeframe]
#    return data_dict


def resample_to_frequency(df, freq):
    # Resample the dataframe to the specified frequency using forward-fill to handle NaNs
    return df.resample(freq).ffill()


# Create sequences and split them for the LSTM
def create_and_split_sequences(
    data_dict, input_length, validation_pct, test_pct, base_freq="D"
):
    # Resample all timeframes to the base frequency of 15 minutes
    resampled_data = {
        tf: resample_to_frequency(df, base_freq) for tf, df in data_dict.items()
    }

    # Align lengths by truncating to the shortest length after resampling
    min_length = min(len(df) for df in resampled_data.values())
    aligned_data = {
        tf: df.iloc[:min_length].reset_index(drop=True)
        for tf, df in resampled_data.items()
    }

    # Concatenate data from all timeframes
    concatenated_data = pd.concat(aligned_data.values(), axis=1)

    # Create sequences
    num_sequences = len(concatenated_data) - input_length + 1
    X = np.zeros(
        (num_sequences, input_length, concatenated_data.shape[1]), dtype=np.float32
    )
    # Array for last observations
    last_observations = np.zeros(
        (num_sequences, 4), dtype=np.float32  # only storing OHLC
    )

    for i in range(num_sequences):
        X[i] = concatenated_data.iloc[i : (i + input_length)].values
        # Capture the last observation (close, high, low)
        last_observations[i] = concatenated_data.iloc[i + input_length - 1][
            [0, 1, 2, 3]
        ].values

    # Split the data
    n = X.shape[0]
    test_index = int(n * (1 - test_pct))
    validation_index = int(n * (1 - test_pct - validation_pct))

    train_X = X[:validation_index]
    val_X = X[validation_index:test_index]
    test_X = X[test_index:]

    # Split the last observations data
    train_last_obs = last_observations[:validation_index]
    val_last_obs = last_observations[validation_index:test_index]
    test_last_obs = last_observations[test_index:]

    return (
        train_X.astype(np.float32),
        val_X.astype(np.float32),
        test_X.astype(np.float32),
        train_last_obs.astype(np.float32),
        val_last_obs.astype(np.float32),
        test_last_obs.astype(np.float32),
    )


# Same sequence function with action-state information


def create_and_split_sequences_static(
    data_dict,
    input_length,
    validation_pct,
    test_pct,
    base_freq="D",
    num_action_state_features=14,  # Number of action-state features to include
):
    # Resample all timeframes to the base frequency of 15 minutes
    resampled_data = {
        tf: resample_to_frequency(df, base_freq) for tf, df in data_dict.items()
    }

    # Align lengths by truncating to the shortest length after resampling
    min_length = min(len(df) for df in resampled_data.values())
    aligned_data = {
        tf: df.iloc[:min_length].reset_index(drop=True)
        for tf, df in resampled_data.items()
    }

    # Concatenate data from all timeframes
    concatenated_data = pd.concat(aligned_data.values(), axis=1)

    # Add placeholders for action-state features to each sequence
    concatenated_data = pd.concat(
        [
            concatenated_data,
            pd.DataFrame(
                np.zeros((len(concatenated_data), num_action_state_features)),
                columns=[f"action_state_{i}" for i in range(num_action_state_features)],
            ),
        ],
        axis=1,
    )

    # Create sequences
    num_sequences = len(concatenated_data) - input_length + 1
    X = np.zeros(
        (num_sequences, input_length, concatenated_data.shape[1]), dtype=np.float32
    )
    for i in range(num_sequences):
        X[i] = concatenated_data.iloc[i : (i + input_length)].values

    # Split the data
    n = X.shape[0]
    test_index = int(n * (1 - test_pct))
    validation_index = int(n * (1 - test_pct - validation_pct))

    train_X = X[:validation_index]
    val_X = X[validation_index:test_index]
    test_X = X[test_index:]

    return (
        train_X.astype(np.float32),
        val_X.astype(np.float32),
        test_X.astype(np.float32),
    )


# A simple function that just returns the batch as is.
def process_batch(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return batch


def convert_to_ray_datasets(train_X, val_X, test_X, batch_size):
    # Assuming train_X, val_X, test_X are numpy arrays coming from your sequence creation function

    # Convert numpy arrays to Ray datasets
    train_dataset = from_numpy(train_X)
    val_dataset = from_numpy(val_X)
    test_dataset = from_numpy(test_X)

    # Batch the datasets
    train_batches = train_dataset.map_batches(process_batch, batch_size=batch_size)
    val_batches = val_dataset.map_batches(process_batch, batch_size=batch_size)
    test_batches = test_dataset.map_batches(process_batch, batch_size=batch_size)

    train_batches = train_batches.materialize()

    return train_batches, val_batches, test_batches


def convert_to_torch_datasets(train_X, val_X, test_X, batch_size):
    # Assuming train_X, val_X, test_X are numpy arrays coming from your sequence creation function

    # Convert numpy arrays to Ray datasets
    train_dataset = torch.tensor(train_X, dtype=torch.float32)
    val_dataset = torch.tensor(val_X, dtype=torch.float32)
    test_dataset = torch.tensor(test_X, dtype=torch.float32)

    # Batch the datasets
    train_batches = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    val_batches = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_batches = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_batches, val_batches, test_batches


def initialize_static_dataset(num_batches, num_features):
    # Initialize the dataset with zeros
    static_data = np.zeros((num_batches, num_features), dtype=np.float32)
    return static_data


def convert_static_to_ray_dataset(static_data, batch_size):
    # Convert the static numpy array to a Ray dataset
    static_dataset = ray.data.from_numpy(static_data)
    # Batch the dataset
    static_batches = static_dataset.map_batches(process_batch, batch_size=batch_size)
    static_batches = static_batches.materialize()
    return static_batches


# Function to dynamically adjust prefetch_batches
def adjust_prefetch(dataset: Dataset, batch_size: int, memory_fraction=0.25):
    # Retrieve Ray's allocated memory (assuming Ray has been initialized)
    ray_memory = 6 * (1024**3)

    # Estimate memory usage per batch (assuming some average size per observation)
    memory_per_state = dataset.size_bytes() / dataset.count()
    estimated_batch_memory = memory_per_state * batch_size

    # Calculate optimal prefetch_batches
    optimal_prefetch = int((ray_memory * memory_fraction) / estimated_batch_memory)
    return max(1, optimal_prefetch)  # Ensure at least one batch is prefetched


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # Get the total system memory
    # total_memory = psutil.virtual_memory().total

    # Calculate 50% of total system memory
    # memory_to_allocate = total_memory * 0.5

    from_time = "2019-10-04"
    to_time = "2024-04-13"
    symbol = "BTCUSDT"

    # Convert times
    from_time = int(
        datetime.datetime.strptime(from_time, "%Y-%m-%d").timestamp() * 1000
    )
    to_time = int(datetime.datetime.strptime(to_time, "%Y-%m-%d").timestamp() * 1000)

    # Define timeframes
    timeframes = ["1d"]
    dataframes = {}

    for tf in timeframes:
        dataframes[tf] = get_timeframe_data(symbol, from_time, to_time, tf)

    # Syncronize the timeframes after computing the features
    for tf in timeframes:
        dataframes[tf] = calculate_indicators(dataframes[tf]).dropna()

    latest_start_date = None
    earliest_end_date = None

    for df in dataframes.values():
        start_date = df.index.min()
        end_date = df.index.max()
        if latest_start_date is None or start_date > latest_start_date:
            latest_start_date = start_date
        if earliest_end_date is None or end_date < earliest_end_date:
            earliest_end_date = end_date

    # Ensure all DataFrames start and end on these dates
    for tf in dataframes:
        dataframes[tf] = dataframes[tf][
            (dataframes[tf].index >= latest_start_date)
            & (dataframes[tf].index <= earliest_end_date)
        ]

    # Normalize the dataframes and add identifiers in timeframes for the LSTM
    normalized_dataframes, ohlc_scaler = normalize_dataframes(dataframes)
    # normalized_dataframes = add_timeframe_identifier(normalized_dataframes)

    # Sequence and split the normalized data for the LSTM
    input_length = 40  # Define the length of the input window
    validation_pct = 0  # 0% validation set
    test_pct = 0.1  # 10% test set

    batch_size = 1  # Define your batch size based on your system's memory capacity and model requirements
    train_X, val_X, test_X, _, _, _ = create_and_split_sequences(
        normalized_dataframes, input_length, validation_pct, test_pct
    )

    # train_torch_ds, val_torch_ds, test_torch_ds = convert_to_torch_datasets(
    #     train_X, val_X, test_X, batch_size=batch_size
    # )

    train_ray_ds, val_ray_ds, test_ray_ds = convert_to_ray_datasets(
        train_X, val_X, test_X, batch_size
    )  # Now, train_batches, val_batches, and test_batches are Ray datasets that are batched and ready for training.

    num_batches = len(train_X)
    num_features = 14
    static_ds = initialize_static_dataset(num_batches, num_features)

    static_ray_ds = convert_static_to_ray_dataset(static_ds, batch_size)

    print(len(train_X))

    # Save the dataset to a file
    np.save("train_data.npy", train_X)
    np.save("normalized_val_data.npy", val_X)
    np.save("normalized_test_data.npy", test_X)

    from test_trade_env_numpy import TradingEnvironment

    # prefetch_batches = adjust_prefetch(train_ray_ds, batch_size=batch_size)

    print(train_X.shape)

    # Delete the intermediate variables
    del (
        dataframes,
        normalized_dataframes,
        ohlc_scaler,
        train_X,
        val_X,
        test_X,
        static_ds,
        static_ray_ds,
    )

    # Explicitly collect garbage
    gc.collect()

    # Ensure Ray is properly initialized
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, object_store_memory=2 * (1024**3))

    # Register the environment in gymnasium
    register(
        id="trading_test_env_numpy-v0",
        entry_point="test_trade_env_numpy:TradingEnvironment",
    )

    # Creates the environment for the ray library
    def env_creator(env_config):
        logger.debug(f"Creating environment with config: {env_config}")
        return TradingEnvironment(
            input_length=env_config.get("input_length", 40),
        )

    # Define the environment creator function
    # def env_creator(env_config):
    #     return TradingEnvironment(**env_config)

    # Register the custom environment
    register_env("trading_test_env_numpy-v0", env_creator)

    # Configuration using PPOConfig
    config = PPOConfig().environment(
        env="trading_test_env_numpy-v0",
        env_config={
            "input_length": 40,
        },
    )
    config.framework("torch")
    config.resources(num_gpus=1, num_cpus_per_worker=3)
    config.rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=360,
        batch_mode="complete_episodes",
    )
    config.training(
        gamma=0.99,
        lr=5e-2,
        train_batch_size=1440,
        sgd_minibatch_size=200,
        num_sgd_iter=30,
        shuffle_sequences=False,
        grad_clip=0.5,
        lambda_=0.99,
        entropy_coeff=0.003,
        clip_param=0.2,
    )
    # Access the model configuration directly via the `.model` attribute
    config.model["use_lstm"] = True
    config.model["lstm_cell_size"] = 32
    config.model["fcnet_hiddens"] = [16]
    config.model["fcnet_activation"] = "relu"
    config.model["post_fcnet_activation"] = "linear"
    config.model["lstm_use_prev_action_reward"] = True
    config.model["max_seq_len"] = 40
    config.model["_disable_action_flattening"] = True

    # results = tune.run(
    #     "PPO",
    #     config=config,
    #     metric="episode_reward_mean",
    #     mode="max",
    #     stop={"training_iteration": 3},
    #     # num_samples=1,  # Number of different sets of hyperparameters to try
    #     search_alg=basic_variant.BasicVariantGenerator(),  # Simple random search
    #     # scheduler=scheduler,
    #     verbose=1,
    #     # checkpoint_freq=50,  # Save a checkpoint every 10 training iterations
    #     checkpoint_at_end=True,  # Ensure a checkpoint is saved at the end of training
    #     local_dir=r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_1ère\2ème_semestre\Advanced Data Analytics\Project\RL_checkpoints",
    # )

    test_env = gymnasium.make("trading_test_env_numpy-v0", input_length=40)
    observation_space = test_env.observation_space
    action_space = test_env.action_space

    # # Define the paths
    base_dir = r"C:\Users\marko\ray_results\PPO_2024-07-07_15-56-10\PPO_trading_env_dynamic_numpy-v0_aa33f_00000_0_2024-07-07_15-56-10\checkpoint_000010"
    # params_path = os.path.join(base_dir, "params.json")
    # checkpoint_path = os.path.join(
    #     base_dir, "checkpoint_000004", "rllib_checkpoint.json"
    # )

    # # Load the params
    # with open(params_path, "r") as f:
    #     checkpoint_config = json.load(f)

    # # Ensure 'normalized_data' is in the config
    # checkpoint_config["env_config"][
    #     "normalized_data"
    # ] = test_ray_ds  # Replace with actual test dataset
    # checkpoint_config["env"] = "trading_env_default-v0"

    # # Ensure 'callbacks' is properly set
    # checkpoint_config["callbacks"] = DefaultCallbacks

    # # Ensure 'policy_mapping_fn' is a callable
    # if "multiagent" in checkpoint_config:
    #     checkpoint_config["multiagent"][
    #         "policy_mapping_fn"
    #     ] = lambda agent_id: "default_policy"
    # else:
    #     checkpoint_config["multiagent"] = {
    #         "policy_mapping_fn": lambda agent_id: "default_policy"
    #     }

    # # Remove or correctly configure the `sample_collector` and `sampler` fields if they exist
    # checkpoint_config.pop("sample_collector", None)
    # checkpoint_config.pop("sampler", None)

    # # Build the trainer
    # trainer_class = get_trainable_cls("PPO")
    # trainer = trainer_class(env="trading_env_default-v0", config=checkpoint_config)

    trainer = config.build()

    # Restore the checkpoint
    trainer.restore(base_dir)

    # Verify the checkpoint has been loaded correctly
    print(f"Restored from checkpoint: {base_dir}")

    state, _ = test_env.reset()
    terminated = False
    cumulative_reward = 0

    # Initial LSTM state (empty state)
    lstm_state = trainer.get_policy().get_initial_state()
    # Print the initial LSTM state
    print(f"INITIAL LSTM STATE: {lstm_state}")
    max_steps = 50  # Set the maximum number of steps to print
    step_count = 0

    while not terminated:

        if state.shape != (input_length, state.shape[1]):
            raise ValueError(f"Unexpected state shape: {state.shape}")

        # Convert the state to a tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        action, lstm_state, _ = trainer.compute_single_action(
            state_tensor,
            state=lstm_state,
            explore=False,
        )

        # if step_count <= max_steps:
        #     print(
        #         f"Step: {test_env.current_step}, Action: {action}, LSTM State: {lstm_state}"
        #     )

        state, reward, terminated, truncated, info = test_env.step(action)
        cumulative_reward += reward

        # if step_count <= max_steps:
        #     print(
        #         f"Reward: {reward}, Cumulative Reward: {cumulative_reward}, Terminated: {terminated}"
        #     )

        step_count += 1

    print(f"Total reward on test data: {cumulative_reward}")
    # After the testing simulation loop in your main script
    test_env.render()
