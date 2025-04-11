import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime
from typing import Dict
import gc
import gymnasium

from database import Hdf5client
import utils
from trade_env_default import TradingEnvironment

from tqdm import tqdm
import time

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search import basic_variant
from ray.data import Dataset, from_numpy


from torch.utils.data import Dataset, DataLoader
import torch


# Data retrieving
def get_timeframe_data(symbol, from_time, to_time, timeframe):
    h5_db = Hdf5client("binance")
    data = h5_db.get_data(symbol, from_time, to_time)
    if timeframe != "1m":
        data = utils.resample_timeframe(data, timeframe)
    return data


def prepare_additional_data(file_path, asset_prefix, timeframe):
    """
    Prepares additional data in the same format as the EURUSD example and resamples it
    to match the provided timeframe.

    Parameters
    ----------
    file_path : str
        The path to the CSV file.
    asset_prefix : str
        The prefix to prepend to column names, e.g. 'eurusd' or 'ustbond'.
    timeframe : str
        The target timeframe to which the data should be resampled (e.g., '4h', '1h', etc.).

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by timestamp at the specified timeframe and columns renamed
        with the asset_prefix.
    """
    # Read the CSV
    df = pd.read_csv(file_path)

    # Convert the timestamp from milliseconds to datetime
    df["timestamp"] = pd.to_datetime(df["Local time"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Keep only the required columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Rename columns to include the asset prefix
    df.columns = [f"{asset_prefix}_{col.lower()}" for col in df.columns]

    # The original data is in 1m timeframe by default, so resample if needed
    if timeframe != "1m":
        df = utils.resample_timeframe(df, timeframe)

    return df


from_time = "2019-11-01"
to_time = "2024-09-01"
symbol = "BTCUSDT"

# Convert times
from_time = int(datetime.datetime.strptime(from_time, "%Y-%m-%d").timestamp() * 1000)
to_time = int(datetime.datetime.strptime(to_time, "%Y-%m-%d").timestamp() * 1000)

data = get_timeframe_data(symbol, from_time, to_time, "1h")
ethusdt_df = get_timeframe_data("ETHUSDT", from_time, to_time, "1h")
# Rename columns to include the asset prefix
ethusdt_df.columns = [f"ethusdt_{col.lower()}" for col in ethusdt_df.columns]
# Additional data preparation and resampling to match main_data timeframe
uk100_df = prepare_additional_data(
    "data/UK100/uk100_bid_cleaned.csv", "uk100", timeframe="1h"
)

# Merge all into a single DataFrame
final_data = data.join(ethusdt_df, how="left").join(uk100_df, how="left")
final_data = final_data.ffill().bfill()  # If you want to fill missing values

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(final_data)
