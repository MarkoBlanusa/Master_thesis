import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import os
import random
import math
from ray.data import read_parquet, Dataset, from_numpy
from functools import partial
import itertools
import ray


# project_dir = r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_1ère\2ème_semestre\ADA_2.0\Source_code_ADA"
# hardcoded_dataset = np.load(os.path.join(project_dir, "train_data.npy"))


# (TO DO : MODIFY THE MARGIN MANAGEMENT)
# (TO DO : CHECK THE ADJUSTED STOP LOSSES FOR THE CASES WHERE MIN IS NEGATIVE BUT MAX IS POSITIVE)
# (TO DO : CHECK THE OPENING FEES BECAUSE OF THE DYNAMIC LEVERAGE)

# (TO DO : THINK ABOUT ADDING A MINIMUM VALUE FOR EACH TRADE)
# (TO DO : ADD AN ESTIMATION OF THE BID AND ASK PRICES)
# TO DO : THINK IF ADDING THE FEES INSIDE THE POSITION VALUE IS GOOD OR NOT
# (TO DO : INCORPORATE THE MINIMUM PRICE MOVEMENT TO BE CONSIDERED PROFIT. IF UNDER CONSIDER IT AS NO PRICE MOVEMENT).
# (TO DO : INCORPORATE THE MAXIMUM ORDER SIZE FOR LIMIT AND MARKET ORDERS.)
# (TO DO : INCORPORATE THE MINIMUM TRADE AMOUNT)
# (TO DO : INCORPORATE A CAP AND FLOOR RATE FOR THE LIMIT ORDERS)
# (TO DO : ADD THE PRICE PROTECTION THRESHOLD) NOT POSSIBLE
# TO DO : ADD THE MIN BTC VALUE IN USDT FOR THE LIMIT ORDERS
# TO DO : ADD A TAKE PROFIT MINIMUM ABOVE THE ENTRY PRICE IN PERCENTAGE

# (TO DO : CORRECT THE MARGIN PRICE MANAGEMENT)


class TradingEnvironment(gym.Env):
    def __init__(
        self,
        input_length=40,
        market_fee=0.0005,
        limit_fee=0.0002,
        liquidation_fee=0.0125,
        slippage_mean=0.000001,
        slippage_std=0.00005,
        initial_balance=1000,
        total_episodes=1001,
        episode_length=168,  # 3 hours of minutes data
        max_risk=0.02,
        min_risk=0.001,
        min_profit=0,
        limit_bounds=False,
    ):
        super(TradingEnvironment, self).__init__()
        self.data = read_parquet("train_dataset_40_1h")
        # self.data_numpy = hardcoded_dataset
        print("LEN DATA : ", self.data.count())
        print("LEN DATA - EPISODE LENGTH : ", self.data.count() - episode_length)
        self.input_length = input_length
        self.max_leverage = 125
        self.leverage = 1
        self.market_fee = market_fee
        self.limit_fee = limit_fee
        self.liquidation_fee = liquidation_fee
        self.slippage_mean = slippage_mean
        self.slippage_std = slippage_std
        self.initial_balance = initial_balance
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.unrealized_pnl = 0
        self.desired_position_size = 0
        self.realized_pnl = 0
        self.position_value = 0
        self.episode_length = episode_length
        self.compounded_returns = 1.0
        self.opening_fee = 0
        self.closing_fee = 0
        self.max_risk = max_risk
        self.min_risk = min_risk
        self.min_profit = min_profit
        self.first_tier_size = 50000
        self.min_trade_btc_amount = 0.001
        self.max_open_orders = 200
        self.min_size_usdt = 100
        self.price_precision = 0.01
        self.max_market_btc_amount = 120
        self.max_limit_btc_amount = 1000
        self.mark_price_rate = 0.00025
        self.cap_rate = 0.05
        self.floor_rate = 0.05
        self.min_price_change_usdt = 0.1
        self.min_btc_value_usdt = 556.8
        self.refill_rate = 0.05
        self.technical_miss = 0.0002
        self.no_trade = 0.0001
        self.max_technical_miss = 0.002
        self.max_no_trade = 0.001
        self.consecutive_technical_miss = 0
        self.consecutive_no_trade = 0
        self.previous_max_dd = 0
        self.take_profit_price = 0
        self.stop_loss_price = 0
        self.entry_price = 0
        self.balance = self.initial_balance
        self.allowed_leverage = 1
        self.margin_fee = 0
        self.current_ask = 0
        self.current_bid = 0
        self.mark_price = 0
        self.log_trading_returns = []
        self.profits = []
        self.losses = []
        self.limit_bounds = limit_bounds
        self.margin_price = 0
        self.current_position_size = 0
        self.current_risk = 0
        self.risk_adjusted_step = 0
        self.sharpe_ratio = 0
        self.new_margin_price = 0
        self.previous_leverage = 1

        if self.limit_bounds:
            # Define action space: weight, stop_loss, take_profit, leverage
            self.action_space = spaces.Box(
                low=np.array([-1, 0, 0, 0]),
                high=np.array([1, 1, 1, 1]),
                dtype=np.float32,
            )
        else:
            # Define action space: weight, leverage
            self.action_space = spaces.Box(
                low=np.array([-1, 0]),
                high=np.array([1, 1]),
                dtype=np.float32,
            )

        # Initialize metrics
        if not hasattr(self, "metrics"):
            self.metrics = {
                "returns": [],
                "num_margin_calls": [],
                "risk_taken": [],
                "sharpe_ratios": [],
                "drawdowns": [],
                "num_trades": [],
                "leverage_used": [],
                "final_balance": [],
                "compounded_returns": [],
                "log_returns": [],
                "profit_risk_ratio": [],
                "sharpe_ratio": [],
                "rewards": [],
            }

        self.sequence_buffer = []
        # self.sequence_buffer_numpy = []
        self.state, info = self.reset()

        if self.limit_bounds:

            self.default_static_values = np.array(
                [
                    self.unrealized_pnl,
                    self.realized_pnl,
                    self.compounded_returns - 1,
                    self.take_profit_price,
                    self.stop_loss_price,
                    self.entry_price,
                    self.leverage,
                    self.allowed_leverage,
                    self.balance,
                    self.position_value,
                    self.desired_position_size,
                    self.current_position_size,
                    0,
                    self.previous_max_dd,
                    self.closing_fee,
                    self.opening_fee,
                    self.current_ask,
                    self.current_bid,
                    self.mark_price,
                    self.margin_price,
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
                    self.new_margin_price,
                ]
            )

        else:

            self.default_static_values = np.array(
                [
                    self.unrealized_pnl,
                    self.realized_pnl,
                    self.compounded_returns - 1,
                    self.entry_price,
                    self.leverage,
                    self.allowed_leverage,
                    self.balance,
                    self.position_value,
                    self.desired_position_size,
                    self.current_position_size,
                    0,
                    self.previous_max_dd,
                    self.closing_fee,
                    self.opening_fee,
                    self.current_ask,
                    self.current_bid,
                    self.mark_price,
                    self.margin_price,
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
                    self.new_margin_price,
                ]
            )

        num_features = self.state.shape[1]

        # Define observation space with appropriate dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.input_length, num_features),
            dtype=np.float32,
        )

    def reset(self, seed=None, **kwargs):

        super().reset(seed=seed, **kwargs)  # Call to super to handle seeding properly

        data_length = self.data.count()
        max_start_index = data_length - self.episode_length
        print("DATA LENGTH FROM RESET : ", data_length)
        print("MAX START INDEX FROM RESET : ", (max_start_index - 1))
        if max_start_index <= 0:
            raise ValueError("Dataset is too small for the specified episode_length.")

        # Randomly select a starting point in the dataset, ensuring there's enough data left for the episode
        self.start_idx = random.randint(0, max_start_index - 1)
        print("START INDEX FROM RESET : ", self.start_idx)
        self.current_step = self.start_idx

        self.unrealized_pnl = 0
        self.position_value = 0
        self.desired_position_size = 0
        self.compounded_returns = 1.0
        self.margin_call_triggered = False
        self.balance = self.initial_balance
        self.portfolio_value = self.balance
        self.positions = []
        self.history = [self.balance]
        self.trading_returns = []  # Initialize trading returns
        self.log_trading_returns = []
        self.final_returns = []
        self.stop_loss_levels = {}
        self.take_profit_levels = {}
        self.cumulative_transaction_costs = 0
        self.previous_action = [0, 0, 0, 0]  # Initial dummy action
        self.opening_fee = 0
        self.closing_fee = 0
        self.leverage = 1
        self.consecutive_technical_miss = 0
        self.consecutive_no_trade = 0
        self.previous_max_dd = 0
        self.take_profit_price = 0
        self.stop_loss_price = 0
        self.entry_price = 0
        self.allowed_leverage = 1
        self.margin_fee = 0
        self.current_ask = 0
        self.current_bid = 0
        self.mark_price = 0
        self.log_trading_returns = []
        self.sequence_buffer = []
        self.profits = []
        self.losses = []
        self.margin_price = 0
        self.current_position_size = 0
        self.current_risk = 0
        self.risk_adjusted_step = 0
        self.sharpe_ratio = 0
        self.new_margin_price = 0
        self.previous_leverage = 1
        # self.sequence_buffer_numpy = []

        # Initialize episode-specific metrics
        self.episode_metrics = {
            "returns": [],
            "compounded_returns": [],
            "num_margin_calls": 0,
            "list_margin_calls": [],
            "risk_taken": [],
            "sharpe_ratios": [],
            "drawdowns": [],
            "num_trades": 0,
            "list_trades": [],
            "leverage_used": [],
            "stop_loss_hits": 0,
            "rewards": [],
            "sharpe_ratio": [],
            "profit_risk_ratio": [],
            "log_returns": [],
            "risk_adjusted_step_return": [],
            "rolling_sharpe": [],
            "step_risk": [],
        }

        self.current_episode += 1

        if self.limit_bounds:

            self.default_static_values = np.array(
                [
                    self.unrealized_pnl,
                    self.realized_pnl,
                    self.compounded_returns - 1,
                    self.take_profit_price,
                    self.stop_loss_price,
                    self.entry_price,
                    self.leverage,
                    self.allowed_leverage,
                    self.balance,
                    self.position_value,
                    self.desired_position_size,
                    self.current_position_size,
                    0,
                    self.previous_max_dd,
                    self.closing_fee,
                    self.opening_fee,
                    self.current_ask,
                    self.current_bid,
                    self.mark_price,
                    self.margin_price,
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
                    self.new_margin_price,
                ]
            )

        else:

            self.default_static_values = np.array(
                [
                    self.unrealized_pnl,
                    self.realized_pnl,
                    self.compounded_returns - 1,
                    self.entry_price,
                    self.leverage,
                    self.allowed_leverage,
                    self.balance,
                    self.position_value,
                    self.desired_position_size,
                    self.current_position_size,
                    0,
                    self.previous_max_dd,
                    self.closing_fee,
                    self.opening_fee,
                    self.current_ask,
                    self.current_bid,
                    self.mark_price,
                    self.margin_price,
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
                    self.new_margin_price,
                ]
            )

        self.num_state_features = len(self.default_static_values)

        # Prepare the windowed dataset for the episode
        self.end_idx = self.start_idx + self.episode_length

        # Split the dataset at self.start_idx
        datasets = self.data.split_at_indices([self.start_idx])
        # Use the dataset starting from self.start_idx
        sliced_data = datasets[1]

        # Initialize the iterator over the windowed dataset
        self.iterator = iter(
            sliced_data.iter_batches(
                batch_size=1,  # Each data point is a pre-sequenced array
                prefetch_batches=self.episode_length,
                drop_last=False,
            )
        )

        # self.iterator = itertools.islice(self.iterator, self.start_idx, None)

        self.sequence_buffer.clear()
        self.load_initial_sequences()
        # self.sequence_buffer_numpy.clear()
        # self.load_initial_sequences_numpy()

        # Ensure the sequence buffer is not empty before accessing
        if len(self.sequence_buffer) == 0:
            raise ValueError(
                "Sequence buffer is empty after loading initial sequences."
            )

        self.state = self.sequence_buffer[0]
        # self.state_numpy = self.sequence_buffer_numpy[0]

        return self.state, {}

    def load_initial_sequences(self):
        """Initialize the sequence buffer with the initial sequences for the rolling window."""

        self.sequence_buffer.clear()

        for _ in range(self.input_length):
            try:
                batch = next(self.iterator)
                initial_data = batch["data"][0].copy()
                initial_data[:, -self.num_state_features :] = self.default_static_values
                self.sequence_buffer.append(initial_data)
            except StopIteration:
                break  # In case there are fewer batches than the input_length

    def update_sequences(self, new_data):
        # new_data should contain the updated last 14 features for the most current observation
        # This method will cascade this new_data through the last 14 features of each sequence's corresponding observation
        # Handle the addition of a completely new sequence and retire the oldest sequence
        try:
            batch = next(self.iterator)
            new_sequence = batch["data"][0].copy()
            # Replace the oldest sequence with a new one if available
            self.sequence_buffer.append(new_sequence)
            self.sequence_buffer.pop(0)
            # Update the last row's last 18 features in each sequence appropriately
            for i in range(len(self.sequence_buffer)):
                # Update the (length-i-1)'th observation in the i'th sequence in the buffer
                self.sequence_buffer[i][-i - 1, -self.num_state_features :] = new_data
            next_state = self.sequence_buffer[0]
            self.state = next_state

            # Check if the episode has ended
            terminated = self.current_step >= self.end_idx
            return next_state, terminated, {}

        except StopIteration:

            print("STOP ITERATION ENCOUNTERED ")
            print("CURRENT STEP FROM STOPITERATION : ", self.current_step)
            print("REMAINING SEQUENCE BUFFER LENGTH : ", len(self.sequence_buffer))
            print(
                "REMAINING STEPS UNTIL END OF EPISODE : ",
                (self.end_idx - self.current_step),
            )
            print("DATA LENGTH : ", self.data.count())
            print(
                "MAX STARTING POINT : ", (self.data.count() - self.episode_length - 1)
            )

            # Even if no more sequences are left for the iterator, finish the remaining sequences inside the buffer
            if len(self.sequence_buffer) > 1:
                self.sequence_buffer.pop(0)
                for i in range(len(self.sequence_buffer)):
                    self.sequence_buffer[i][
                        -i - 1, -self.num_state_features :
                    ] = new_data
                next_state = self.sequence_buffer[0]
                self.state = next_state
                terminated = self.current_step >= self.end_idx
                return next_state, terminated, {}

            else:
                # Reset the episode if the batch buffer ends
                # self.get_aggregated_trade_info()
                # next_state, info = self.reset()
                self.state = self.sequence_buffer[0]
                terminated = True
                return self.state, terminated, {}

    # def load_initial_sequences_numpy(self):
    #     """Initialize the sequence buffer based on the current episode starting index."""
    #     self.sequence_buffer_numpy.clear()

    #     # Load the initial sequence from the starting point in the dataset
    #     for i in range(self.input_length):
    #         initial_data_idx = self.start_idx + i
    #         if initial_data_idx < (self.start_idx + self.episode_length):
    #             initial_data = self.data_numpy[initial_data_idx]
    #             # Replace the last 18 columns with default static values
    #             initial_data[:, -18:] = self.default_static_values
    #             self.sequence_buffer_numpy.append(initial_data)
    #         else:
    #             break  # Stop if we've reached the episode length limit

    # def update_sequences_numpy(self, new_data):
    #     """Update the sequence buffer as the episode progresses, adding new data and popping old data."""

    #     # Normalize the new data (as needed)
    #     normalized_new_data = new_data

    #     # Check if the next step exceeds the episode length
    #     if (self.current_step + self.input_length) < (
    #         self.start_idx + self.episode_length
    #     ):
    #         # Append the new sequence from the dataset if within episode bounds
    #         new_sequence_idx = self.current_step + self.input_length
    #         new_sequence = self.data_numpy[new_sequence_idx]
    #         self.sequence_buffer_numpy.append(new_sequence)
    #         self.sequence_buffer_numpy.pop(
    #             0
    #         )  # Remove the oldest sequence from the buffer
    #     else:
    #         self.sequence_buffer_numpy.pop(
    #             0
    #         )  # Only remove from the buffer if no new sequence is available

    #     # Update the last 18 columns of the sequences in the buffer with the new data
    #     if len(self.sequence_buffer_numpy) > 0:
    #         for i in range(len(self.sequence_buffer_numpy)):
    #             self.sequence_buffer_numpy[i][-i - 1, -18:] = normalized_new_data

    #         # Set the next state to be the first sequence in the buffer
    #         next_state = self.sequence_buffer_numpy[0]
    #         self.state_numpy = next_state
    #         self.current_step += 1
    #         terminated = self.current_step >= (self.start_idx + self.episode_length)

    #         return next_state, terminated, {}

    #     else:
    #         # If the buffer is empty, reset the environment (end of episode)
    #         # next_state, info = self.reset()
    #         self.state_numpy = next_state
    #         terminated = True
    #         return next_state, terminated, {}

    #     # if len(self.sequence_buffer) == 1:
    #     #     self.render()

    #     # if len(self.sequence_buffer) <= 0:
    #     #     next_state, info = self.reset()
    #     #     terminated = True
    #     #     return next_state, terminated, info

    def get_std_dev_from_volume(
        self,
        volume,
        min_std=0.001,
        max_std=0.01,
        scaling_factor=7000,
        fallback_std=0.005,
    ):

        # Handle NaN or zero volume cases
        if np.isnan(volume) or volume == 0:
            return fallback_std

        # Calculate the inverse volume effect
        raw_std_dev = 1 / (volume / scaling_factor)

        # Normalize to a range between min_std and max_std
        normalized_std_dev = min_std + (max_std - min_std) * (
            raw_std_dev / (1 + raw_std_dev)
        )

        # If normalized std_dev is NaN or inf, fallback to fixed std_dev
        if np.isnan(normalized_std_dev) or np.isinf(normalized_std_dev):
            return fallback_std

        return normalized_std_dev

    def approximate_bid_ask(
        self,
        high_price,
        low_price,
        close_price,
        volume,
        bid_ask_std_base=0.0015,
        scaling_factor=1000,
        fallback_std_dev=0.0025,  # Fixed std_dev for fallback mechanism
    ):

        range_price = high_price - low_price

        # Check for NaN in high, low, or volume to prevent NaN results
        if (
            np.isnan(high_price)
            or np.isnan(low_price)
            or np.isnan(volume)
            or volume == 0
        ):
            # Use fallback method if inputs are invalid
            return self.fallback_approximation(close_price, fallback_std_dev)

        # Adjust std_dev based on volume
        std_dev = bid_ask_std_base / (volume / scaling_factor)

        # Check if std_dev is NaN or infinity
        if np.isnan(std_dev) or np.isinf(std_dev):
            return self.fallback_approximation(close_price, fallback_std_dev)

        bid_spread = np.random.normal(0, std_dev) * range_price
        ask_spread = np.random.normal(0, std_dev) * range_price

        bid_price = close_price - bid_spread
        ask_price = close_price + ask_spread

        # Check if bid_price and ask_price is NaN or infinity
        if (
            np.isnan(bid_price)
            or np.isnan(ask_price)
            or np.isinf(bid_price)
            or np.isinf(ask_price)
        ):
            return self.fallback_approximation(close_price, fallback_std_dev)

        return bid_price, ask_price

    def fallback_approximation(self, current_price, fixed_std_dev):
        """
        Fallback method to approximate bid and ask prices if NaN is encountered.
        Uses the current_price and applies fixed bid/ask spreads based on a fixed std_dev.
        """
        range_price = current_price * 0.01  # Assume a 1% range for spread approximation

        # Generate fixed spreads using a normal distribution with fixed_std_dev
        bid_spread = np.random.normal(0, fixed_std_dev) * range_price
        ask_spread = np.random.normal(0, fixed_std_dev) * range_price

        # Calculate fallback bid and ask prices
        bid_price = current_price - bid_spread
        ask_price = current_price + ask_spread

        return bid_price, ask_price

    def step(self, action):

        print("CURRENT STEP BEGINNING OF STEP FUNCTION : ", self.current_step)

        if self.limit_bounds:

            weight, stop_loss, take_profit, leverage = action

            # # Generate random values for each parameter

            # # weight: Random float between -1 and 1
            # weight = np.random.uniform(-1, 1)

            # # stop_loss: Random float between 0 and 1
            # stop_loss = np.random.uniform(0, 1)

            # # take_profit: Random float between 0 and 1
            # take_profit = np.random.uniform(0, 1)

            # # leverage: Random integer between 1 and 125
            # leverage = np.random.uniform(0, 1)

            # # Now weight, stop_loss, take_profit, and leverage are randomly generated

        else:

            weight, leverage = action

            # # Generate random values for each parameter

            # # weight: Random float between -1 and 1
            # weight = np.random.uniform(-1, 1)

            # # leverage: Random integer between 1 and 125
            # leverage = np.random.uniform(0, 1)

            # # Now weight and leverage are randomly generated

        # weight, stop_loss, take_profit, leverage = action
        self.previous_action = action

        # Normalize the leverage between 1 and 125
        leverage = leverage * (self.max_leverage - 1) + 1

        # Make the leverage an integer value
        leverage = round(leverage)

        # Define a small tolerance value
        epsilon = 1e-10

        # Get the latest observation from the current sequence for price logic
        current_price = self.state[-1, 3]
        current_high = self.state[-1, 1]
        current_low = self.state[-1, 2]
        current_open = self.state[-1, 0]
        current_volume = self.state[-1, 4]

        # current_price_npy = self.state_numpy[-1, 3]
        # current_high_npy = self.state_numpy[-1, 1]
        # current_low_npy = self.state_numpy[-1, 2]
        # current_open_npy = self.state_numpy[-1, 0]
        # current_volume_npy = self.state_numpy[-1, 4]

        # print("CURRENT STATE'S LAST OBSERVATION : ", self.state[-1])

        # with np.printoptions(threshold=np.inf):
        #     print("STATE FROM STEP FUNCTION : ", self.state)
        #     # print("STATE FROM NUMPY : ", self.state_numpy)

        print("CURRENT OPEN : ", current_open)
        print("CURRENT HIGH : ", current_high)
        print("CURRENT LOW : ", current_low)
        print("CURRENT CLOSE : ", current_price)
        print()
        # print("CURRENT OPEN NUMPY : ", current_open_npy)
        # print("CURRENT HIGH NUMPY : ", current_high_npy)
        # print("CURRENT LOW NUMPY : ", current_low_npy)
        # print("CURRENT CLOSE NUMPY : ", current_price_npy)
        print()
        # print("SEQUENCE BUFFER : ", self.sequence_buffer)

        # Approximate the bid and ask prices
        current_bid, current_ask = self.approximate_bid_ask(
            current_high, current_low, current_price, current_volume
        )
        self.current_bid, self.current_ask = current_bid, current_ask

        # Round the bid and ask prices
        rounded_bid = (
            round(current_bid / self.min_price_change_usdt) * self.min_price_change_usdt
        )
        rounded_ask = (
            round(current_ask / self.min_price_change_usdt) * self.min_price_change_usdt
        )

        # Round the close price when executed for Binance standard
        rounded_price = (
            round(current_price / self.min_price_change_usdt)
            * self.min_price_change_usdt
        )

        # Approximate a mark price simulation
        mark_mean = 0
        mark_std = self.get_std_dev_from_volume(current_volume)
        normal_distrib_factor = np.random.normal(mark_mean, mark_std)
        mark_price = current_price * (1 + normal_distrib_factor)
        self.mark_price = mark_price

        # Reset the realized pnl
        self.realized_pnl = 0

        # Reset the margin fee
        self.margin_fee = 0

        # Total value of positions and current weight
        if self.positions:
            self.position_value = (sum(p[2] for p in self.positions)) / self.positions[
                -1
            ][3]
            current_weight = sum(
                [pos[2] * (1 if pos[0] == "long" else -1) for pos in self.positions]
            ) / ((self.balance + self.position_value) * self.positions[-1][3])
            self.previous_leverage = self.positions[-1][3]

        else:
            self.position_value = 0
            current_weight = 0
            self.previous_leverage = 1

        print("LEVERAGE OUTPUTED : ", leverage)
        print("WEIGHT OUTPUTED : ", weight)
        print("CURRENT WEIGHT: ", current_weight)
        weight_diff = weight - current_weight
        print("WEIGHT DIFF: ", weight_diff)

        # Apply slippage to the execution price
        slippage = np.random.normal(self.slippage_mean, self.slippage_std)

        effective_bid_price = current_bid * (
            (1 + slippage) if weight_diff > 0 else (1 - slippage)
        )
        effective_ask_price = current_ask * (
            (1 + slippage) if weight_diff > 0 else (1 - slippage)
        )
        effective_market_bid_price = (
            round(effective_bid_price / self.min_price_change_usdt)
            * self.min_price_change_usdt
        )
        effective_market_ask_price = (
            round(effective_ask_price / self.min_price_change_usdt)
            * self.min_price_change_usdt
        )

        reward = 0

        # Force to liquidate all position at the end of the episode
        if (self.current_step + 1) >= (
            self.start_idx + self.episode_length
        ) and self.positions:
            # Entire position is closed

            print("CURRENT STEP END OF EPISODE : ", self.current_step)
            print(
                "START INDEX + EPISODE LENGTH : ",
                (self.start_idx + self.episode_length),
            )

            # Compute the realized pnl
            if self.positions[-1][0] == "long":
                realized_pnl = self.positions[-1][2] * (
                    (effective_market_bid_price - self.positions[-1][1])
                    / self.positions[-1][1]
                )
            else:
                realized_pnl = self.positions[-1][2] * (
                    (self.positions[-1][1] - effective_market_ask_price)
                    / self.positions[-1][1]
                )

            # Update the balance
            self.closing_fee = self.positions[-1][2] * self.market_fee
            self.balance += (realized_pnl - self.closing_fee - self.opening_fee) + (
                self.positions[-1][2] / self.positions[-1][3]
            )

            self.realized_pnl = realized_pnl - self.closing_fee - self.opening_fee

            # Update the unrealized pnl
            self.unrealized_pnl = 0

            # Update the fees
            self.closing_fee = 0
            self.opening_fee = 0

            # The entire position is closed, remove it
            self.positions = []  # Remove closed positions
            self.position_value = 0
            print("LAST STEP OF EPISODE REACHED, CLOSING THE POSITION ")
            print("REALIZED PNL OF THE LAST STEP : ", self.realized_pnl)

        else:

            # Perform margin checks before attempting to open new positions
            self.check_margin_call(current_high, current_low)
            if self.limit_bounds:
                # Check for stop-loss, take-profit
                self.check_limits(current_high, current_low)

            # Calculate the desired position size
            desired_position_size = (
                weight * (self.balance + self.position_value) * leverage
            )
            print("DESIRED POSITION SIZE : ", desired_position_size)
            self.desired_position_size = desired_position_size

            # Check the position size to see if it fits the margin rules and adapt the optimal leverage
            allowed_leverage, _, _, lower_size, upper_size = self.get_margin_tier(
                abs(desired_position_size)
            )
            self.allowed_leverage = allowed_leverage
            print("LOWER SIZE: ", lower_size)
            print("UPPER SIZE: ", upper_size)
            print("ALLOWED LEVERAGE : ", allowed_leverage)

            if leverage > allowed_leverage:
                penalty = min(
                    self.technical_miss
                    + (0.5 * self.technical_miss * self.consecutive_technical_miss),
                    self.max_technical_miss,
                )
                reward -= penalty
                self.consecutive_technical_miss += 1

                print("PENALTY FROM LEVERAGE : ", penalty)
                print("LEVERAGE TOO HIGH FOR THE DESIRED SIZE. APPLYING PENALTY. ")

                # Check if there is an open position
                if self.positions:

                    # Update the unrealized pnl
                    self.unrealized_pnl = (
                        self.positions[-1][2]
                        * (
                            (rounded_bid - self.positions[-1][1])
                            / self.positions[-1][1]
                        )
                        if self.positions[-1][0] == "long"
                        else self.positions[-1][2]
                        * (
                            (self.positions[-1][1] - rounded_ask)
                            / self.positions[-1][1]
                        )
                    )

                    # Check if limit orders mode is activated
                    if self.limit_bounds:

                        # Check if having a stop loss under the risk level is feasible
                        max_loss_per_unit = (
                            self.max_risk * (self.balance + self.position_value)
                            - (2 * self.opening_fee)
                        ) / self.positions[-1][2]
                        min_loss_per_unit = (
                            self.min_risk * (self.balance + self.position_value)
                            - (2 * self.opening_fee)
                        ) / self.positions[-1][2]

                        # Restricted loss
                        restricted_loss_per_unit = 1 - (
                            (1 - self.cap_rate) * (mark_price / self.positions[-1][1])
                        )

                        # Compute the max
                        final_max_loss_per_unit = min(
                            max_loss_per_unit, restricted_loss_per_unit
                        )

                        print("STOP LOSS WITH OUTPUT : ", stop_loss)
                        print("MAX LOSS PER UNIT : ", max_loss_per_unit)
                        print("MIN LOSS PER UNIT : ", min_loss_per_unit)
                        adjusted_stop_loss = max(
                            min(stop_loss, final_max_loss_per_unit), min_loss_per_unit
                        )
                        print("ADJUSTED STOP LOSS : ", adjusted_stop_loss)

                        # Set new limits even if the position is the same
                        self.set_limits(
                            self.positions[-1][1],
                            take_profit,
                            self.positions[-1][0],
                            adjusted_stop_loss,
                            mark_price,
                        )

                        print("NEW LIMITS APPLIED EVEN IF THE LEVERAGE ISN'T FITTING. ")
            else:
                # Save the leverage if it is fitting the margin criterias
                self.leverage = leverage

                # TO DO : ADAPT THE OPENING FEES
                # if self.max_leverage > allowed_leverage:
                #     max_leverage = math.floor(
                #         self.first_tier_size
                #         / ((self.balance + self.position_value) * abs(weight))
                #     )
                #     self.leverage = max(max_leverage, 1)
                #     desired_position_size = (
                #         weight * (self.balance + self.position_value) * self.leverage
                #     )
                # else:
                #     desired_position_size = desired_max_position_size

                print("TOTAL POSITION SIZE : ", [pos[2] for pos in self.positions])

                # position_size = (
                #     abs(weight_diff) * (self.balance + self.position_value) * self.leverage
                # )
                max_position_size = (self.balance + self.position_value) * self.leverage

                # Determine current position type and set variables accordingly
                current_position_type = (
                    "short" if any(p[0] == "short" for p in self.positions) else "long"
                )

                # Calculate the current position size based on the type
                current_position_size = sum(
                    p[2] * (1 if p[0] == "long" else -1)
                    for p in self.positions
                    if p[0] == current_position_type
                )
                self.current_position_size = current_position_size

                print(
                    "DESIRED POSITION SIZE BEFORE MAIN STEP : ", desired_position_size
                )
                print("BALANCE BEFORE MAIN STEP : ", self.balance)
                print("POSITION VALUE BEFORE MAIN STEP : ", self.position_value)

                # Calculate the difference in size based on the position type
                difference_size = desired_position_size - current_position_size
                print("DIFFERENCE SIZE FIRST : ", difference_size)

                # Compute the desired size in BTC
                desired_btc_size = abs(desired_position_size) / current_price

                print(
                    "STOP LOSS LEVELS AT THE BEGINNING OF STEP : ",
                    self.stop_loss_levels,
                )

                if (
                    abs(desired_position_size) >= self.min_size_usdt
                    and desired_btc_size >= self.min_trade_btc_amount
                    and desired_btc_size <= self.max_market_btc_amount
                ):

                    print(
                        "MINIMUM TRADE AMOUNT AND MINIMUM USDT AMOUNT ARE SATISFIED. "
                    )
                    print("DESIRED POSITION SIZE : ", desired_position_size)
                    print("DESIRED BTC SIZE : ", desired_btc_size)

                    if difference_size > 0:  # Increase position

                        # Increase the long position
                        if current_position_size > 0:

                            difference_size_bound = abs(difference_size)
                            print("POSITION SIZE CHANGE : ", difference_size_bound)

                            increasing_size = difference_size_bound
                            increasing_btc_size = increasing_size / current_price

                            if (
                                increasing_size >= self.min_size_usdt
                                and increasing_btc_size >= self.min_trade_btc_amount
                                and increasing_btc_size <= self.max_market_btc_amount
                            ):

                                required_margin = increasing_size / self.leverage
                                (
                                    position_type,
                                    entry_price,
                                    current_size,
                                    previous_leverage,
                                ) = self.positions[-1]
                                print(
                                    "REQUIRED MARGIN FOR THE NEW LONG POSITION : ",
                                    required_margin,
                                )
                                if self.balance >= (required_margin - epsilon):

                                    # Combine with existing long position
                                    new_size = current_size + increasing_size
                                    new_entry_price = entry_price * (
                                        current_size / new_size
                                    ) + effective_market_ask_price * (
                                        increasing_size / new_size
                                    )
                                    combined_leverage = new_size / (
                                        self.position_value + required_margin
                                    )

                                    # # Combine with existing long position
                                    # for i, pos in enumerate(self.positions):
                                    #     if pos[0] == "long":
                                    #         new_size = pos[2] + increasing_size
                                    #         new_entry_price = pos[1] * (
                                    #             pos[2] / new_size
                                    #         ) + effective_price * (increasing_size / new_size)
                                    #         # self.positions[i] = ("long", new_entry_price, new_size)
                                    #         # print(f"UPDATED LONG POSITION: {self.positions[i]}")
                                    #         break

                                    # Calculate and deduct the transaction fee
                                    self.opening_fee += (
                                        increasing_size * self.market_fee
                                    )

                                    if self.limit_bounds:

                                        # Check if having a stop loss under the risk level is feasible
                                        max_loss_per_unit = (
                                            self.max_risk
                                            * (self.balance + self.position_value)
                                            - (2 * self.opening_fee)
                                        ) / new_size
                                        min_loss_per_unit = (
                                            self.min_risk
                                            * (self.balance + self.position_value)
                                            - (2 * self.opening_fee)
                                        ) / new_size

                                        # Restricted loss
                                        restricted_loss_per_unit = 1 - (
                                            (1 - self.cap_rate)
                                            * (mark_price / self.positions[-1][1])
                                        )

                                        # Compute the max
                                        final_max_loss_per_unit = min(
                                            max_loss_per_unit, restricted_loss_per_unit
                                        )

                                        print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                        print(
                                            "RESTRICTED LOSS PER UNIT : ",
                                            restricted_loss_per_unit,
                                        )
                                        print("MAX LOSS PER UNIT : ", max_loss_per_unit)
                                        print("MIN LOSS PER UNIT : ", min_loss_per_unit)
                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
                                        )
                                        print(
                                            "ADJUSTED STOP LOSS : ", adjusted_stop_loss
                                        )

                                        if (
                                            adjusted_stop_loss > 0
                                            and adjusted_stop_loss >= min_loss_per_unit
                                        ):

                                            # Update the balance
                                            self.balance -= required_margin
                                            print(
                                                "BALANCE CHANGE (REQUIRED MARGIN FOR INCREASING LONG POSITION): ",
                                                -required_margin,
                                            )

                                            # Update the new increased position
                                            self.positions[-1] = (
                                                "long",
                                                new_entry_price,
                                                new_size,
                                                combined_leverage,
                                            )
                                            print(
                                                f"UPDATED LONG POSITION: {self.positions[-1]}"
                                            )

                                            # Update the unrealized pnl
                                            self.unrealized_pnl = new_size * (
                                                (
                                                    effective_market_bid_price
                                                    - new_entry_price
                                                )
                                                / new_entry_price
                                            )
                                            self.unrealized_pnl -= self.opening_fee
                                            print(
                                                "ENTRY PRICE AND EFFECTIVE AREN'T THE SAME BECAUSE WE INCREASED THE POSITION. "
                                            )

                                            # Set the limits
                                            self.set_limits(
                                                new_entry_price,
                                                take_profit,
                                                "long",
                                                adjusted_stop_loss,
                                                mark_price,
                                            )

                                            # Update the cumulative transactions costs
                                            self.cumulative_transaction_costs += (
                                                increasing_size * self.market_fee
                                            )

                                            # Update the metrics
                                            leverage_used = (
                                                new_size / max_position_size
                                            ) * combined_leverage
                                            self.episode_metrics[
                                                "leverage_used"
                                            ].append(leverage_used)
                                            self.episode_metrics["num_trades"] += 1
                                            self.episode_metrics["list_trades"].append(
                                                self.episode_metrics["num_trades"]
                                            )

                                            self.consecutive_technical_miss = 0
                                            self.consecutive_no_trade = 0

                                            print("NEW POSITION CREATED FOR A NEW LONG")

                                        else:
                                            # Put back the opening fees
                                            self.opening_fee -= (
                                                increasing_size * self.market_fee
                                            )

                                            # Check if having a stop loss under the risk level is feasible
                                            max_loss_per_unit = (
                                                self.max_risk
                                                * (self.balance + self.position_value)
                                                - (2 * self.opening_fee)
                                            ) / current_size
                                            min_loss_per_unit = (
                                                self.min_risk
                                                * (self.balance + self.position_value)
                                                - (2 * self.opening_fee)
                                            ) / current_size

                                            # Restricted loss
                                            restricted_loss_per_unit = 1 - (
                                                (1 - self.cap_rate)
                                                * (mark_price / self.positions[-1][1])
                                            )

                                            # Compute the max
                                            final_max_loss_per_unit = min(
                                                max_loss_per_unit,
                                                restricted_loss_per_unit,
                                            )

                                            print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                            print(
                                                "RESTRICTED LOSS PER UNIT : ",
                                                restricted_loss_per_unit,
                                            )
                                            print(
                                                "MAX LOSS PER UNIT : ",
                                                max_loss_per_unit,
                                            )
                                            print(
                                                "MIN LOSS PER UNIT : ",
                                                min_loss_per_unit,
                                            )
                                            adjusted_stop_loss = max(
                                                min(stop_loss, final_max_loss_per_unit),
                                                min_loss_per_unit,
                                            )

                                            print(
                                                "ADJUSTED STOP LOSS : ",
                                                adjusted_stop_loss,
                                            )

                                            # Set new limits even if the position is the same
                                            self.set_limits(
                                                entry_price,
                                                take_profit,
                                                "long",
                                                adjusted_stop_loss,
                                                mark_price,
                                            )

                                            # Update the unrealized pnl
                                            self.unrealized_pnl = current_size * (
                                                (rounded_bid - entry_price)
                                                / entry_price
                                            )

                                            # Penalize the model
                                            penalty = min(
                                                self.technical_miss
                                                + (
                                                    0.5
                                                    * self.technical_miss
                                                    * self.consecutive_technical_miss
                                                ),
                                                self.max_technical_miss,
                                            )
                                            reward -= penalty
                                            self.consecutive_technical_miss += 1

                                            print("PENALTY FROM FEES : ", penalty)

                                            print(
                                                "WE DON'T INCREASE THE LONG POSITION BECAUSE THE RISK TOLERANCE IS NOT FEASIBLE WITH THE FEES. "
                                            )
                                    else:
                                        # Update the balance
                                        self.balance -= required_margin
                                        print(
                                            "BALANCE CHANGE (REQUIRED MARGIN FOR INCREASING LONG POSITION): ",
                                            -required_margin,
                                        )

                                        # Update the new increased position
                                        self.positions[-1] = (
                                            "long",
                                            new_entry_price,
                                            new_size,
                                            combined_leverage,
                                        )
                                        print(
                                            f"UPDATED LONG POSITION: {self.positions[-1]}"
                                        )

                                        # Update the unrealized pnl
                                        self.unrealized_pnl = new_size * (
                                            (
                                                effective_market_bid_price
                                                - new_entry_price
                                            )
                                            / new_entry_price
                                        )
                                        self.unrealized_pnl -= self.opening_fee
                                        print(
                                            "ENTRY PRICE AND EFFECTIVE AREN'T THE SAME BECAUSE WE INCREASED THE POSITION. "
                                        )

                                        # Update the cumulative transactions costs
                                        self.cumulative_transaction_costs += (
                                            increasing_size * self.market_fee
                                        )

                                        # Update the metrics
                                        leverage_used = (
                                            new_size / max_position_size
                                        ) * combined_leverage
                                        self.episode_metrics["leverage_used"].append(
                                            leverage_used
                                        )
                                        self.episode_metrics["num_trades"] += 1
                                        self.episode_metrics["list_trades"].append(
                                            self.episode_metrics["num_trades"]
                                        )

                                        self.consecutive_technical_miss = 0
                                        self.consecutive_no_trade = 0

                                        print("NEW POSITION CREATED FOR A NEW LONG")
                                        print("NO RISK LIMIT MODE")

                                else:
                                    if self.limit_bounds:
                                        # Check if having a stop loss under the risk level is feasible
                                        max_loss_per_unit = (
                                            self.max_risk
                                            * (self.balance + self.position_value)
                                            - (2 * self.opening_fee)
                                        ) / current_size
                                        min_loss_per_unit = (
                                            self.min_risk
                                            * (self.balance + self.position_value)
                                            - (2 * self.opening_fee)
                                        ) / current_size

                                        # Restricted loss
                                        restricted_loss_per_unit = 1 - (
                                            (1 - self.cap_rate)
                                            * (mark_price / self.positions[-1][1])
                                        )

                                        # Compute the max
                                        final_max_loss_per_unit = min(
                                            max_loss_per_unit, restricted_loss_per_unit
                                        )

                                        print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                        print(
                                            "RESTRICTED LOSS PER UNIT : ",
                                            restricted_loss_per_unit,
                                        )
                                        print("MAX LOSS PER UNIT : ", max_loss_per_unit)
                                        print("MIN LOSS PER UNIT : ", min_loss_per_unit)
                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
                                        )
                                        print(
                                            "ADJUSTED STOP LOSS : ", adjusted_stop_loss
                                        )

                                        # Set new limits even if the position is the same
                                        self.set_limits(
                                            entry_price,
                                            take_profit,
                                            "long",
                                            adjusted_stop_loss,
                                            mark_price,
                                        )

                                    # Update the unrealized pnl
                                    self.unrealized_pnl = current_size * (
                                        (rounded_bid - entry_price) / entry_price
                                    )

                                    # Penalize the model
                                    penalty = min(
                                        self.technical_miss
                                        + (
                                            0.5
                                            * self.technical_miss
                                            * self.consecutive_technical_miss
                                        ),
                                        self.max_technical_miss,
                                    )
                                    reward -= penalty
                                    self.consecutive_technical_miss += 1

                                    print("PENALTY FROM INITIAL MARGIN : ", penalty)

                                    print(
                                        "NOT ENOUGH BALANCE TO INCREASE THE NEW LONG POSITION. CURRENT BALANCE : ",
                                        self.balance,
                                    )

                            else:
                                # Penalize the model
                                penalty = min(
                                    self.technical_miss
                                    + (
                                        0.5
                                        * self.technical_miss
                                        * self.consecutive_technical_miss
                                    ),
                                    self.max_technical_miss,
                                )
                                reward -= penalty
                                self.consecutive_technical_miss += 1

                                print("PENALTY FROM MINIMUM SIZE : ", penalty)

                                print(
                                    "INCREASING SIZE TOO SHORT OR TOO BIG : ",
                                    increasing_size,
                                )

                        # Diminishing the short position and potentitally closing it to open a long position
                        if current_position_size < 0:  # Closing a short position
                            print("CURRENT WEIGHT IS NEGATIVE. ")
                            # Close the short position
                            # closing_size = min(
                            #     position_size, sum(p[2] for p in self.positions if p[0] == "short")
                            # )

                            difference_size = abs(difference_size)
                            print("POSITION SIZE CHANGE : ", difference_size)

                            # Ensure to not close more than the current short position
                            closing_size = min(
                                difference_size, abs(current_position_size)
                            )

                            print(
                                "PRICE AT WHICH THE WEIGHT WAS UPDATED : ",
                                effective_market_ask_price,
                            )

                            # If the entire position is not closed, update the remaining position
                            if (closing_size + epsilon) < abs(current_position_size):

                                closing_btc_size = closing_size / current_price

                                if (
                                    closing_size >= self.min_size_usdt
                                    and closing_btc_size >= self.min_trade_btc_amount
                                    and closing_btc_size <= self.max_market_btc_amount
                                ):

                                    remaining_size = (
                                        abs(current_position_size) - closing_size
                                    )
                                    print("REMAINING SIZE : ", remaining_size)

                                    # Execute the buy order to close part of the short position
                                    realized_pnl = closing_size * (
                                        (
                                            self.positions[0][1]
                                            - effective_market_ask_price
                                        )
                                        / self.positions[0][1]
                                    )

                                    # Update the opening fee
                                    self.opening_fee -= closing_size * self.market_fee

                                    # Compute the closing fee
                                    self.closing_fee = closing_size * self.market_fee

                                    if self.limit_bounds:

                                        # Check if having a stop loss under the risk level is feasible
                                        max_loss_per_unit = (
                                            self.max_risk
                                            * (
                                                (
                                                    self.balance
                                                    + realized_pnl
                                                    - 2 * self.closing_fee
                                                    + (
                                                        closing_size
                                                        / self.positions[-1][3]
                                                    )
                                                )
                                                + (
                                                    remaining_size
                                                    / self.positions[-1][3]
                                                )
                                            )
                                            - (2 * self.opening_fee)
                                        ) / remaining_size
                                        min_loss_per_unit = (
                                            self.min_risk
                                            * (
                                                (
                                                    self.balance
                                                    + realized_pnl
                                                    - 2 * self.closing_fee
                                                    + (
                                                        closing_size
                                                        / self.positions[-1][3]
                                                    )
                                                )
                                                + (
                                                    remaining_size
                                                    / self.positions[-1][3]
                                                )
                                            )
                                            - (2 * self.opening_fee)
                                        ) / remaining_size

                                        # Restricted loss
                                        restricted_loss_per_unit = 1 - (
                                            (1 - self.cap_rate)
                                            * (mark_price / self.positions[-1][1])
                                        )

                                        # Compute the max
                                        final_max_loss_per_unit = min(
                                            max_loss_per_unit, restricted_loss_per_unit
                                        )

                                        print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                        print(
                                            "RESTRICTED LOSS PER UNIT : ",
                                            restricted_loss_per_unit,
                                        )
                                        print("MAX LOSS PER UNIT : ", max_loss_per_unit)
                                        print("MIN LOSS PER UNIT : ", min_loss_per_unit)
                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
                                        )
                                        print(
                                            "ADJUSTED STOP LOSS : ", adjusted_stop_loss
                                        )

                                        if (
                                            adjusted_stop_loss > 0
                                            and adjusted_stop_loss >= min_loss_per_unit
                                        ):

                                            self.positions[0] = (
                                                self.positions[0][0],
                                                self.positions[0][1],
                                                remaining_size,
                                                self.positions[0][3],
                                            )
                                            print(
                                                "PARTIAL POSITION REMAINS: ",
                                                self.positions[0],
                                            )
                                            print("WEIGHT_DIFF : ", weight_diff)

                                            # Calculate and deduct the transaction fee
                                            self.balance += (
                                                realized_pnl - 2 * self.closing_fee
                                            ) + (closing_size / self.positions[0][3])

                                            self.realized_pnl = (
                                                realized_pnl - 2 * self.closing_fee
                                            )

                                            print(
                                                "NET PROFIT FROM DIMINISHING THE SHORT POSITION : ",
                                                (realized_pnl - 2 * self.closing_fee),
                                            )
                                            self.cumulative_transaction_costs += (
                                                closing_size * self.market_fee
                                            )

                                            # Update the unrealized pnl
                                            self.unrealized_pnl = remaining_size * (
                                                (
                                                    self.positions[0][1]
                                                    - effective_market_ask_price
                                                )
                                                / self.positions[0][1]
                                            )
                                            self.unrealized_pnl -= self.opening_fee

                                            # Update the limit orders
                                            self.set_limits(
                                                self.positions[0][1],
                                                take_profit,
                                                "short",
                                                adjusted_stop_loss,
                                                mark_price,
                                            )

                                            # Update the fees
                                            self.closing_fee = 0

                                            self.consecutive_technical_miss = 0
                                            self.consecutive_no_trade = 0

                                            print(
                                                "ENTRY PRICE AND EFFECTIVE PRICE AREN'T THE SAME BECAUSE WE PARTIALLY CLOSED THE SHORT POSITION. "
                                            )

                                        else:
                                            # If the remaining size doesn't allow a valid stop loss, keep the initial position

                                            # Put back the opening fees
                                            self.opening_fee += (
                                                closing_size * self.market_fee
                                            )

                                            # Check if having a stop loss under the risk level is feasible
                                            max_loss_per_unit = (
                                                self.max_risk
                                                * (self.balance + self.position_value)
                                                - (2 * self.opening_fee)
                                            ) / self.positions[-1][2]
                                            min_loss_per_unit = (
                                                self.min_risk
                                                * (self.balance + self.position_value)
                                                - (2 * self.opening_fee)
                                            ) / self.positions[-1][2]

                                            # Restricted loss
                                            restricted_loss_per_unit = 1 - (
                                                (1 - self.cap_rate)
                                                * (mark_price / self.positions[-1][1])
                                            )

                                            # Compute the max
                                            final_max_loss_per_unit = min(
                                                max_loss_per_unit,
                                                restricted_loss_per_unit,
                                            )

                                            print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                            print(
                                                "RESTRICTED LOSS PER UNIT : ",
                                                restricted_loss_per_unit,
                                            )
                                            print(
                                                "MAX LOSS PER UNIT : ",
                                                max_loss_per_unit,
                                            )
                                            print(
                                                "MIN LOSS PER UNIT : ",
                                                min_loss_per_unit,
                                            )
                                            adjusted_stop_loss = max(
                                                min(stop_loss, final_max_loss_per_unit),
                                                min_loss_per_unit,
                                            )
                                            print(
                                                "ADJUSTED STOP LOSS : ",
                                                adjusted_stop_loss,
                                            )

                                            # Set new limits even if the position is the same
                                            self.set_limits(
                                                self.positions[-1][1],
                                                take_profit,
                                                "short",
                                                adjusted_stop_loss,
                                                mark_price,
                                            )

                                            # Update the unrealized pnl
                                            self.unrealized_pnl = self.positions[-1][
                                                2
                                            ] * (
                                                (self.positions[-1][1] - rounded_ask)
                                                / self.positions[-1][1]
                                            )

                                            # Penalize the model
                                            penalty = min(
                                                self.technical_miss
                                                + (
                                                    0.5
                                                    * self.technical_miss
                                                    * self.consecutive_technical_miss
                                                ),
                                                self.max_technical_miss,
                                            )
                                            reward -= penalty
                                            self.consecutive_technical_miss += 1

                                            print("PENALTY FROM FEES : ", penalty)

                                            # # Execute the buy order to close the rest of the position
                                            # realized_pnl = remaining_size * (
                                            #     (self.positions[0][1] - effective_price)
                                            #     / self.positions[0][1]
                                            # )

                                            # # Calculate and deduct the transaction fee
                                            # self.closing_fee = remaining_size * self.market_fee
                                            # self.balance += (
                                            #     realized_pnl - self.closing_fee - self.opening_fee
                                            # ) + (remaining_size / self.leverage)

                                            # self.cumulative_transaction_costs += (
                                            #     remaining_size * self.market_fee
                                            # )

                                            # # Close the entire position
                                            # self.positions = [
                                            #     p for p in self.positions if p[0] != "short"
                                            # ]  # Remove closed positions

                                            # # Update the unrealized pnl
                                            # self.unrealized_pnl = 0

                                            # # Update the fees
                                            # self.opening_fee = 0
                                            # self.closing_fee = 0

                                            # # Penalize the model
                                            # reward -= 500

                                            # print(
                                            #     "ENTIRE POSITION IS CLOSED BECAUSE THE RISK TOLERANCE WASN'T MET AFTER THE DECREASE OF THE SHORT POSITION, WEIGHT_DIFF : ",
                                            #     weight_diff,
                                            # )

                                    else:

                                        self.positions[0] = (
                                            self.positions[0][0],
                                            self.positions[0][1],
                                            remaining_size,
                                            self.positions[0][3],
                                        )
                                        print(
                                            "PARTIAL POSITION REMAINS: ",
                                            self.positions[0],
                                        )
                                        print("WEIGHT_DIFF : ", weight_diff)

                                        # Calculate and deduct the transaction fee
                                        self.balance += (
                                            realized_pnl - 2 * self.closing_fee
                                        ) + (closing_size / self.positions[0][3])

                                        self.realized_pnl = (
                                            realized_pnl - 2 * self.closing_fee
                                        )

                                        print(
                                            "NET PROFIT FROM DIMINISHING THE SHORT POSITION : ",
                                            (realized_pnl - 2 * self.closing_fee),
                                        )
                                        self.cumulative_transaction_costs += (
                                            closing_size * self.market_fee
                                        )

                                        # Update the unrealized pnl
                                        self.unrealized_pnl = remaining_size * (
                                            (
                                                self.positions[0][1]
                                                - effective_market_ask_price
                                            )
                                            / self.positions[0][1]
                                        )
                                        self.unrealized_pnl -= self.opening_fee

                                        # Update the fees
                                        self.closing_fee = 0

                                        self.consecutive_technical_miss = 0
                                        self.consecutive_no_trade = 0

                                        print(
                                            "ENTRY PRICE AND EFFECTIVE PRICE AREN'T THE SAME BECAUSE WE PARTIALLY CLOSED THE SHORT POSITION. "
                                        )
                                        print("NO MORE RISK LIMIT")

                                else:

                                    # Penalize the model
                                    penalty = min(
                                        self.technical_miss
                                        + (
                                            0.5
                                            * self.technical_miss
                                            * self.consecutive_technical_miss
                                        ),
                                        self.max_technical_miss,
                                    )
                                    reward -= penalty
                                    self.consecutive_technical_miss += 1

                                    print(
                                        "PENALTY FROM CLOSING SIZE BEING TOO SHORT OR TOO BIG : ",
                                        penalty,
                                    )
                                    print("CLOSING SIZE : ", closing_size)

                            else:

                                # Compute the realized pnl
                                realized_pnl = closing_size * (
                                    (self.positions[-1][1] - effective_market_ask_price)
                                    / self.positions[-1][1]
                                )

                                # Update the balance
                                self.closing_fee = closing_size * self.market_fee
                                self.balance += (
                                    realized_pnl - self.closing_fee - self.opening_fee
                                ) + (closing_size / self.positions[0][3])

                                self.realized_pnl = (
                                    realized_pnl - self.closing_fee - self.opening_fee
                                )

                                # Update the unrealized pnl
                                self.unrealized_pnl = 0

                                # Update the fees
                                self.closing_fee = 0
                                self.opening_fee = 0

                                # If the entire position is closed, remove it
                                self.positions = [
                                    p for p in self.positions if p[0] != "short"
                                ]  # Remove closed positions

                                self.consecutive_technical_miss = 0
                                self.consecutive_no_trade = 0

                                print(
                                    "ENTIRE POSITION IS CLOSED IMMEDIATELY, WEIGHT_DIFF : ",
                                    weight_diff,
                                )

                            print(
                                "POSITION SIZE BEFORE CLOSING SIZE : ",
                                current_position_size,
                            )
                            # position_size -= closing_size
                            print(
                                "POSITION SIZE AFTER CLOSING SIZE : ",
                                (current_position_size - closing_size),
                            )
                            print("CLOSING SHORT SIZE : ", closing_size)

                        # Open a new long position with the exceeding size
                        if (
                            current_position_size <= 0
                            and (abs(difference_size) - abs(current_position_size)) > 0
                            and desired_position_size > 0
                        ):
                            new_position_size = abs(difference_size) - abs(
                                current_position_size
                            )
                            print("DIFFERENCE SIZE : ", abs(difference_size))
                            print("CURRENT SIZE : ", current_position_size)
                            print("NEW POSITION SIZE : ", new_position_size)
                            required_margin = new_position_size / self.leverage
                            print(
                                "REQUIRED MARGIN FOR THE NEW LONG POSITION : ",
                                required_margin,
                            )
                            if self.balance >= (required_margin - epsilon):

                                # Calculate and deduct the transcation fee separately
                                self.opening_fee = new_position_size * self.market_fee

                                if self.limit_bounds:

                                    # Check if having a stop loss under the risk level is feasible
                                    max_loss_per_unit = (
                                        self.max_risk
                                        * (self.balance + self.position_value)
                                        - (2 * self.opening_fee)
                                    ) / new_position_size
                                    min_loss_per_unit = (
                                        self.min_risk
                                        * (self.balance + self.position_value)
                                        - (2 * self.opening_fee)
                                    ) / new_position_size

                                    # Restricted loss
                                    restricted_loss_per_unit = 1 - (
                                        (1 - self.cap_rate)
                                        * (mark_price / effective_market_ask_price)
                                    )

                                    # Compute the max
                                    final_max_loss_per_unit = min(
                                        max_loss_per_unit, restricted_loss_per_unit
                                    )

                                    print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                    print(
                                        "RESTRICTED LOSS PER UNIT : ",
                                        restricted_loss_per_unit,
                                    )
                                    print("MAX LOSS PER UNIT : ", max_loss_per_unit)
                                    print("MIN LOSS PER UNIT : ", min_loss_per_unit)
                                    adjusted_stop_loss = max(
                                        min(stop_loss, final_max_loss_per_unit),
                                        min_loss_per_unit,
                                    )
                                    print("ADJUSTED STOP LOSS : ", adjusted_stop_loss)

                                    if (
                                        adjusted_stop_loss > 0
                                        and adjusted_stop_loss >= min_loss_per_unit
                                    ):

                                        self.balance -= required_margin
                                        print(
                                            "BALANCE CHANGE (REQUIRED MARGIN NEW LONG POSITION) : ",
                                            -required_margin,
                                        )

                                        self.positions.append(
                                            (
                                                "long",
                                                effective_market_ask_price,
                                                new_position_size,
                                                self.leverage,
                                            )
                                        )

                                        # Update the unrealized pnl
                                        self.unrealized_pnl = new_position_size * (
                                            (rounded_bid - effective_market_ask_price)
                                            / effective_market_ask_price
                                        )
                                        self.unrealized_pnl -= self.opening_fee
                                        print(
                                            "ENTRY PRICE AND EFFECTIVE PRICE ARE THE SAME BECAUSE WE OPENED A NEW POSITION. "
                                        )

                                        # Update the limit orders
                                        self.set_limits(
                                            self.positions[0][1],
                                            take_profit,
                                            "long",
                                            adjusted_stop_loss,
                                            mark_price,
                                        )

                                        # Update the cumulative transactions costs
                                        self.cumulative_transaction_costs += (
                                            new_position_size * self.market_fee
                                        )

                                        # Update the metrics
                                        leverage_used = (
                                            new_position_size / max_position_size
                                        ) * self.leverage
                                        self.episode_metrics["leverage_used"].append(
                                            leverage_used
                                        )
                                        self.episode_metrics["num_trades"] += 1
                                        self.episode_metrics["list_trades"].append(
                                            self.episode_metrics["num_trades"]
                                        )

                                        self.consecutive_technical_miss = 0
                                        self.consecutive_no_trade = 0

                                        print("NEW POSITION CREATED FOR A NEW LONG")

                                    else:
                                        # Put back the opening fees
                                        self.opening_fee -= (
                                            new_position_size * self.market_fee
                                        )

                                        # Penalize the model
                                        penalty = min(
                                            self.technical_miss
                                            + (
                                                0.5
                                                * self.technical_miss
                                                * self.consecutive_technical_miss
                                            ),
                                            self.max_technical_miss,
                                        )
                                        reward -= penalty
                                        self.consecutive_technical_miss += 1

                                        print("PENALTY FROM FEES : ", penalty)

                                        print(
                                            "WE DON'T OPEN THE NEW LONG POSITION BECAUSE THE RISK TOLERANCE IS NOT FEASIBLE WITH THE FEES. "
                                        )

                                else:

                                    self.balance -= required_margin
                                    print(
                                        "BALANCE CHANGE (REQUIRED MARGIN NEW LONG POSITION) : ",
                                        -required_margin,
                                    )

                                    if self.positions:
                                        self.positions[-1] = (
                                            "long",
                                            effective_market_ask_price,
                                            new_position_size,
                                            self.leverage,
                                        )
                                    else:
                                        self.positions.append(
                                            (
                                                "long",
                                                effective_market_ask_price,
                                                new_position_size,
                                                self.leverage,
                                            )
                                        )

                                    # Update the unrealized pnl
                                    self.unrealized_pnl = new_position_size * (
                                        (rounded_bid - effective_market_ask_price)
                                        / effective_market_ask_price
                                    )
                                    self.unrealized_pnl -= self.opening_fee
                                    print(
                                        "ENTRY PRICE AND EFFECTIVE PRICE ARE THE SAME BECAUSE WE OPENED A NEW POSITION. "
                                    )

                                    # Update the cumulative transactions costs
                                    self.cumulative_transaction_costs += (
                                        new_position_size * self.market_fee
                                    )

                                    # Update the metrics
                                    leverage_used = (
                                        new_position_size / max_position_size
                                    ) * self.leverage
                                    self.episode_metrics["leverage_used"].append(
                                        leverage_used
                                    )
                                    self.episode_metrics["num_trades"] += 1
                                    self.episode_metrics["list_trades"].append(
                                        self.episode_metrics["num_trades"]
                                    )

                                    self.consecutive_technical_miss = 0
                                    self.consecutive_no_trade = 0

                                    print("NEW POSITION CREATED FOR A NEW LONG")
                                    print("NO MORE RISK LIMIT")

                            else:
                                # Penalize the model
                                penalty = min(
                                    self.technical_miss
                                    + (
                                        0.5
                                        * self.technical_miss
                                        * self.consecutive_technical_miss
                                    ),
                                    self.max_technical_miss,
                                )
                                reward -= penalty
                                self.consecutive_technical_miss += 1

                                print("PENALTY FROM INITIAL MARGIN : ", penalty)

                                print(
                                    "NOT ENOUGH BALANCE TO OPEN THE NEW LONG POSITION. CURRENT BALANCE : ",
                                    self.balance,
                                )

                    elif difference_size < 0:  # Decrease position

                        # Increase the short position
                        if current_position_size < 0:

                            difference_size_bound = abs(difference_size)
                            print("POSITION SIZE CHANGE : ", difference_size_bound)
                            increasing_size = difference_size_bound
                            increasing_btc_size = increasing_size / current_price

                            if (
                                increasing_size >= self.min_size_usdt
                                and increasing_btc_size >= self.min_trade_btc_amount
                                and increasing_btc_size <= self.max_market_btc_amount
                            ):

                                required_margin = increasing_size / self.leverage
                                (
                                    position_type,
                                    entry_price,
                                    current_size,
                                    previous_leverage,
                                ) = self.positions[-1]
                                print(
                                    "REQUIRED MARGIN FOR THE NEW SHORT POSITION : ",
                                    required_margin,
                                )
                                if self.balance >= (required_margin - epsilon):

                                    # Combine with existing long position
                                    new_size = current_size + increasing_size
                                    new_entry_price = entry_price * (
                                        current_size / new_size
                                    ) + effective_market_bid_price * (
                                        increasing_size / new_size
                                    )
                                    combined_leverage = new_size / (
                                        self.position_value + required_margin
                                    )
                                    # # Combine with existing short position
                                    # for i, pos in enumerate(self.positions):
                                    #     if pos[0] == "short":
                                    #         new_size = pos[2] + increasing_size
                                    #         new_entry_price = pos[1] * (
                                    #             pos[2] / new_size
                                    #         ) + effective_price * (increasing_size / new_size)
                                    #         self.positions[i] = ("short", new_entry_price, new_size)
                                    #         print(f"UPDATED SHORT POSITION: {self.positions[i]}")
                                    #         break

                                    # Calculate and deduct the transcation fee separately
                                    self.opening_fee += (
                                        increasing_size * self.market_fee
                                    )

                                    if self.limit_bounds:

                                        # Check if having a stop loss under the risk level is feasible
                                        max_loss_per_unit = (
                                            self.max_risk
                                            * (self.balance + self.position_value)
                                            - (2 * self.opening_fee)
                                        ) / new_size
                                        min_loss_per_unit = (
                                            self.min_risk
                                            * (self.balance + self.position_value)
                                            - (2 * self.opening_fee)
                                        ) / new_size

                                        # Restricted loss
                                        restricted_loss_per_unit = 1 - (
                                            (1 - self.cap_rate)
                                            * (mark_price / self.positions[-1][1])
                                        )

                                        # Compute the max
                                        final_max_loss_per_unit = min(
                                            max_loss_per_unit, restricted_loss_per_unit
                                        )

                                        print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                        print(
                                            "RESTRICTED LOSS PER UNIT : ",
                                            restricted_loss_per_unit,
                                        )
                                        print("MAX LOSS PER UNIT : ", max_loss_per_unit)
                                        print("MIN LOSS PER UNIT : ", min_loss_per_unit)
                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
                                        )
                                        print(
                                            "ADJUSTED STOP LOSS : ", adjusted_stop_loss
                                        )

                                        if (
                                            adjusted_stop_loss > 0
                                            and adjusted_stop_loss >= min_loss_per_unit
                                        ):

                                            self.balance -= required_margin
                                            print(
                                                "BALANCE CHANGE (REQUIRED MARGIN FOR INCREASING SHORT POSITION): ",
                                                -required_margin,
                                            )

                                            # Update the new increased position
                                            self.positions[-1] = (
                                                "short",
                                                new_entry_price,
                                                new_size,
                                                combined_leverage,
                                            )
                                            print(
                                                f"UPDATED SHORT POSITION: {self.positions[-1]}"
                                            )

                                            # Update the unrealized pnl
                                            self.unrealized_pnl = new_size * (
                                                (
                                                    new_entry_price
                                                    - effective_market_ask_price
                                                )
                                                / new_entry_price
                                            )
                                            self.unrealized_pnl -= self.opening_fee
                                            print(
                                                "ENTRY PRICE AND EFFECTIVE PRICE AREN'T THE SAME BECAUSE WE INCREASED THE POSITION. "
                                            )

                                            # Set the limits
                                            self.set_limits(
                                                new_entry_price,
                                                take_profit,
                                                "short",
                                                adjusted_stop_loss,
                                                mark_price,
                                            )

                                            # Update the cumulative transactions costs
                                            self.cumulative_transaction_costs += (
                                                increasing_size * self.market_fee
                                            )

                                            # Update the metrics
                                            leverage_used = (
                                                new_size / max_position_size
                                            ) * combined_leverage
                                            self.episode_metrics[
                                                "leverage_used"
                                            ].append(leverage_used)
                                            self.episode_metrics["num_trades"] += 1
                                            self.episode_metrics["list_trades"].append(
                                                self.episode_metrics["num_trades"]
                                            )

                                            self.consecutive_technical_miss = 0
                                            self.consecutive_no_trade = 0

                                            print(
                                                "NEW POSITION CREATED FOR A NEW SHORT"
                                            )

                                        else:

                                            # Put back the opening fees
                                            self.opening_fee -= (
                                                increasing_size * self.market_fee
                                            )

                                            # Check if having a stop loss under the risk level is feasible
                                            max_loss_per_unit = (
                                                self.max_risk
                                                * (self.balance + self.position_value)
                                                - (2 * self.opening_fee)
                                            ) / current_size
                                            min_loss_per_unit = (
                                                self.min_risk
                                                * (self.balance + self.position_value)
                                                - (2 * self.opening_fee)
                                            ) / current_size

                                            # Restricted loss
                                            restricted_loss_per_unit = 1 - (
                                                (1 - self.cap_rate)
                                                * (mark_price / self.positions[-1][1])
                                            )

                                            # Compute the max
                                            final_max_loss_per_unit = min(
                                                max_loss_per_unit,
                                                restricted_loss_per_unit,
                                            )

                                            print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                            print(
                                                "RESTRICTED LOSS PER UNIT : ",
                                                restricted_loss_per_unit,
                                            )
                                            print(
                                                "MAX LOSS PER UNIT : ",
                                                max_loss_per_unit,
                                            )
                                            print(
                                                "MIN LOSS PER UNIT : ",
                                                min_loss_per_unit,
                                            )
                                            adjusted_stop_loss = max(
                                                min(stop_loss, final_max_loss_per_unit),
                                                min_loss_per_unit,
                                            )
                                            print(
                                                "ADJUSTED STOP LOSS : ",
                                                adjusted_stop_loss,
                                            )

                                            # Set new limits even if the position is the same
                                            self.set_limits(
                                                entry_price,
                                                take_profit,
                                                "short",
                                                adjusted_stop_loss,
                                                mark_price,
                                            )

                                            # Update the unrealized pnl
                                            self.unrealized_pnl = current_size * (
                                                (entry_price - rounded_ask)
                                                / entry_price
                                            )

                                            # Penalize the model
                                            penalty = min(
                                                self.technical_miss
                                                + (
                                                    0.5
                                                    * self.technical_miss
                                                    * self.consecutive_technical_miss
                                                ),
                                                self.max_technical_miss,
                                            )
                                            reward -= penalty
                                            self.consecutive_technical_miss += 1

                                            print("PENALTY FROM FEES : ", penalty)

                                            print(
                                                "WE DON'T INCREASE THE SHORT POSITION BECAUSE THE RISK TOLERANCE IS NOT FEASIBLE WITH THE FEES. "
                                            )

                                    else:

                                        self.balance -= required_margin
                                        print(
                                            "BALANCE CHANGE (REQUIRED MARGIN FOR INCREASING SHORT POSITION): ",
                                            -required_margin,
                                        )

                                        # Update the new increased position
                                        self.positions[-1] = (
                                            "short",
                                            new_entry_price,
                                            new_size,
                                            combined_leverage,
                                        )
                                        print(
                                            f"UPDATED SHORT POSITION: {self.positions[-1]}"
                                        )

                                        # Update the unrealized pnl
                                        self.unrealized_pnl = new_size * (
                                            (
                                                new_entry_price
                                                - effective_market_ask_price
                                            )
                                            / new_entry_price
                                        )
                                        self.unrealized_pnl -= self.opening_fee
                                        print(
                                            "ENTRY PRICE AND EFFECTIVE PRICE AREN'T THE SAME BECAUSE WE INCREASED THE POSITION. "
                                        )

                                        # Update the cumulative transactions costs
                                        self.cumulative_transaction_costs += (
                                            increasing_size * self.market_fee
                                        )

                                        # Update the metrics
                                        leverage_used = (
                                            new_size / max_position_size
                                        ) * combined_leverage
                                        self.episode_metrics["leverage_used"].append(
                                            leverage_used
                                        )
                                        self.episode_metrics["num_trades"] += 1
                                        self.episode_metrics["list_trades"].append(
                                            self.episode_metrics["num_trades"]
                                        )

                                        self.consecutive_technical_miss = 0
                                        self.consecutive_no_trade = 0

                                        print("NEW POSITION CREATED FOR A NEW SHORT")
                                        print("NO MORE RISK LIMIT")

                                else:

                                    if self.limit_bounds:

                                        # Check if having a stop loss under the risk level is feasible
                                        max_loss_per_unit = (
                                            self.max_risk
                                            * (self.balance + self.position_value)
                                            - (2 * self.opening_fee)
                                        ) / current_size
                                        min_loss_per_unit = (
                                            self.min_risk
                                            * (self.balance + self.position_value)
                                            - (2 * self.opening_fee)
                                        ) / current_size

                                        # Restricted loss
                                        restricted_loss_per_unit = 1 - (
                                            (1 - self.cap_rate)
                                            * (mark_price / self.positions[-1][1])
                                        )

                                        # Compute the max
                                        final_max_loss_per_unit = min(
                                            max_loss_per_unit, restricted_loss_per_unit
                                        )

                                        print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                        print(
                                            "RESTRICTED LOSS PER UNIT : ",
                                            restricted_loss_per_unit,
                                        )
                                        print("MAX LOSS PER UNIT : ", max_loss_per_unit)
                                        print("MIN LOSS PER UNIT : ", min_loss_per_unit)
                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
                                        )
                                        print(
                                            "ADJUSTED STOP LOSS : ", adjusted_stop_loss
                                        )

                                        # Set new limits even if the position is the same
                                        self.set_limits(
                                            entry_price,
                                            take_profit,
                                            "short",
                                            adjusted_stop_loss,
                                            mark_price,
                                        )

                                    # Update the unrealized pnl
                                    self.unrealized_pnl = current_size * (
                                        (entry_price - rounded_ask) / entry_price
                                    )

                                    # Penalize the model
                                    penalty = min(
                                        self.technical_miss
                                        + (
                                            0.5
                                            * self.technical_miss
                                            * self.consecutive_technical_miss
                                        ),
                                        self.max_technical_miss,
                                    )
                                    reward -= penalty
                                    self.consecutive_technical_miss += 1

                                    print("PENALTY FROM INITIAL MARGIN : ", penalty)

                                    print(
                                        "NOT ENOUGH BALANCE TO INCREASE THE NEW SHORT POSITION. CURRENT BALANCE : ",
                                        self.balance,
                                    )

                            else:
                                # Penalize the model
                                penalty = min(
                                    self.technical_miss
                                    + (
                                        0.5
                                        * self.technical_miss
                                        * self.consecutive_technical_miss
                                    ),
                                    self.max_technical_miss,
                                )
                                reward -= penalty
                                self.consecutive_technical_miss += 1

                                print(
                                    "INCREASING SIZE TOO SHORT OR TOO BIG : ",
                                    increasing_size,
                                )

                        # Diminishing the long position and potentitally closing it to open a short position
                        if current_position_size > 0:  # Closing a long position
                            print("CURRENT WEIGHT IS POSITIVE. ")
                            # Close the long position
                            # closing_size = min(
                            #     position_size, sum(p[2] for p in self.positions if p[0] == "long")
                            # )

                            difference_size = abs(difference_size)
                            print("POSITION SIZE CHANGE : ", difference_size)

                            # Ensure to not close more than the current long position
                            closing_size = min(
                                difference_size, abs(current_position_size)
                            )

                            print(
                                "PRICE AT WHICH THE WEIGHT WAS UPDATED : ",
                                effective_market_bid_price,
                            )

                            # If the entire position is not closed, update the remaining position
                            if (closing_size + epsilon) < abs(current_position_size):

                                closing_btc_size = closing_size / current_price

                                if (
                                    closing_size >= self.min_size_usdt
                                    and closing_btc_size >= self.min_trade_btc_amount
                                    and closing_btc_size <= self.max_market_btc_amount
                                ):

                                    # Check if we can afford to keep the remaining position while still respecting the risk tolerance
                                    remaining_size = (
                                        abs(current_position_size) - closing_size
                                    )
                                    print("REMAINING SIZE : ", remaining_size)

                                    realized_pnl = closing_size * (
                                        (
                                            effective_market_bid_price
                                            - self.positions[0][1]
                                        )
                                        / self.positions[0][1]
                                    )

                                    # Update the opening fee
                                    self.opening_fee -= closing_size * self.market_fee

                                    # Compute the closing fee
                                    self.closing_fee = closing_size * self.market_fee

                                    if self.limit_bounds:

                                        # Check if having a stop loss under the risk level is feasible
                                        max_loss_per_unit = (
                                            self.max_risk
                                            * (
                                                (
                                                    self.balance
                                                    + realized_pnl
                                                    - 2 * self.closing_fee
                                                    + (
                                                        closing_size
                                                        / self.positions[-1][3]
                                                    )
                                                )
                                                + (
                                                    remaining_size
                                                    / self.positions[-1][3]
                                                )
                                            )
                                            - (2 * self.opening_fee)
                                        ) / remaining_size
                                        min_loss_per_unit = (
                                            self.min_risk
                                            * (
                                                (
                                                    self.balance
                                                    + realized_pnl
                                                    - 2 * self.closing_fee
                                                    + (
                                                        closing_size
                                                        / self.positions[-1][3]
                                                    )
                                                )
                                                + (
                                                    remaining_size
                                                    / self.positions[-1][3]
                                                )
                                            )
                                            - (2 * self.opening_fee)
                                        ) / remaining_size

                                        # Restricted loss
                                        restricted_loss_per_unit = 1 - (
                                            (1 - self.cap_rate)
                                            * (mark_price / self.positions[-1][1])
                                        )

                                        # Compute the max
                                        final_max_loss_per_unit = min(
                                            max_loss_per_unit, restricted_loss_per_unit
                                        )

                                        print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                        print(
                                            "RESTRICTED LOSS PER UNIT : ",
                                            restricted_loss_per_unit,
                                        )
                                        print("MAX LOSS PER UNIT : ", max_loss_per_unit)
                                        print("MIN LOSS PER UNIT : ", min_loss_per_unit)
                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
                                        )
                                        print(
                                            "ADJUSTED STOP LOSS : ", adjusted_stop_loss
                                        )

                                        if (
                                            adjusted_stop_loss > 0
                                            and adjusted_stop_loss >= min_loss_per_unit
                                        ):

                                            self.positions[0] = (
                                                self.positions[0][0],
                                                self.positions[0][1],
                                                remaining_size,
                                                self.positions[-1][3],
                                            )
                                            print(
                                                "PARTIAL POSITION REMAINS: ",
                                                self.positions[0],
                                            )

                                            # Calculate and deduct the transaction fee
                                            self.balance += (
                                                realized_pnl - 2 * self.closing_fee
                                            ) + (closing_size / self.positions[-1][3])

                                            self.realized_pnl = (
                                                realized_pnl - 2 * self.closing_fee
                                            )

                                            print(
                                                "NET PROFIT FROM DIMINISHING THE LONG POSITION : ",
                                                (realized_pnl - 2 * self.closing_fee),
                                            )
                                            self.cumulative_transaction_costs += (
                                                closing_size * self.market_fee
                                            )

                                            # Update the unrealized pnl
                                            self.unrealized_pnl = remaining_size * (
                                                (
                                                    effective_market_bid_price
                                                    - self.positions[0][1]
                                                )
                                                / self.positions[0][1]
                                            )
                                            self.unrealized_pnl -= self.opening_fee

                                            # Update the limit orders
                                            self.set_limits(
                                                self.positions[0][1],
                                                take_profit,
                                                "long",
                                                adjusted_stop_loss,
                                                mark_price,
                                            )

                                            # Update the fees
                                            self.closing_fee = 0

                                            self.consecutive_technical_miss = 0
                                            self.consecutive_no_trade = 0

                                            print(
                                                "ENTRY PRICE AND EFFECTIVE PRICE AREN'T THE SAME BECAUSE WE PARTIALLY CLOSED THE POSITION. "
                                            )

                                        else:
                                            # If the remaining size doesn't allow a valid stop loss, keep the initial position

                                            # Put back the opening fees
                                            self.opening_fee += (
                                                closing_size * self.market_fee
                                            )

                                            # Check if having a stop loss under the risk level is feasible
                                            max_loss_per_unit = (
                                                self.max_risk
                                                * (self.balance + self.position_value)
                                                - (2 * self.opening_fee)
                                            ) / self.positions[-1][2]
                                            min_loss_per_unit = (
                                                self.min_risk
                                                * (self.balance + self.position_value)
                                                - (2 * self.opening_fee)
                                            ) / self.positions[-1][2]

                                            # Restricted loss
                                            restricted_loss_per_unit = 1 - (
                                                (1 - self.cap_rate)
                                                * (mark_price / self.positions[-1][1])
                                            )

                                            # Compute the max
                                            final_max_loss_per_unit = min(
                                                max_loss_per_unit,
                                                restricted_loss_per_unit,
                                            )

                                            print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                            print(
                                                "RESTRICTED LOSS PER UNIT : ",
                                                restricted_loss_per_unit,
                                            )
                                            print(
                                                "MAX LOSS PER UNIT : ",
                                                max_loss_per_unit,
                                            )
                                            print(
                                                "MIN LOSS PER UNIT : ",
                                                min_loss_per_unit,
                                            )
                                            adjusted_stop_loss = max(
                                                min(stop_loss, final_max_loss_per_unit),
                                                min_loss_per_unit,
                                            )
                                            print(
                                                "ADJUSTED STOP LOSS : ",
                                                adjusted_stop_loss,
                                            )

                                            # Set new limits even if the position is the same
                                            self.set_limits(
                                                self.positions[-1][1],
                                                take_profit,
                                                "long",
                                                adjusted_stop_loss,
                                                mark_price,
                                            )

                                            # Update the unrealized pnl
                                            self.unrealized_pnl = self.positions[-1][
                                                2
                                            ] * (
                                                (rounded_bid - self.positions[-1][1])
                                                / self.positions[-1][1]
                                            )

                                            # Penalize the model
                                            penalty = min(
                                                self.technical_miss
                                                + (
                                                    0.5
                                                    * self.technical_miss
                                                    * self.consecutive_technical_miss
                                                ),
                                                self.max_technical_miss,
                                            )
                                            reward -= penalty
                                            self.consecutive_technical_miss += 1

                                            print("PENALTY FROM FEES : ", penalty)

                                            print(
                                                "ENTIRE POSITION IS KEPT BECAUSE THE RISK TOLERANCE WASN'T MET AFTER THE DECREASE OF THE LONG POSITION, WEIGHT_DIFF : ",
                                                weight_diff,
                                            )

                                    else:

                                        self.positions[0] = (
                                            self.positions[0][0],
                                            self.positions[0][1],
                                            remaining_size,
                                            self.positions[-1][3],
                                        )
                                        print(
                                            "PARTIAL POSITION REMAINS: ",
                                            self.positions[0],
                                        )

                                        # Calculate and deduct the transaction fee
                                        self.balance += (
                                            realized_pnl - 2 * self.closing_fee
                                        ) + (closing_size / self.positions[-1][3])

                                        self.realized_pnl = (
                                            realized_pnl - 2 * self.closing_fee
                                        )

                                        print(
                                            "NET PROFIT FROM DIMINISHING THE LONG POSITION : ",
                                            (realized_pnl - 2 * self.closing_fee),
                                        )
                                        self.cumulative_transaction_costs += (
                                            closing_size * self.market_fee
                                        )

                                        # Update the unrealized pnl
                                        self.unrealized_pnl = remaining_size * (
                                            (
                                                effective_market_bid_price
                                                - self.positions[0][1]
                                            )
                                            / self.positions[0][1]
                                        )
                                        self.unrealized_pnl -= self.opening_fee

                                        # Update the fees
                                        self.closing_fee = 0

                                        self.consecutive_technical_miss = 0
                                        self.consecutive_no_trade = 0

                                        print(
                                            "ENTRY PRICE AND EFFECTIVE PRICE AREN'T THE SAME BECAUSE WE PARTIALLY CLOSED THE POSITION. "
                                        )
                                        print("NO MORE RISK LIMIT")

                                else:

                                    # Penalize the model
                                    penalty = min(
                                        self.technical_miss
                                        + (
                                            0.5
                                            * self.technical_miss
                                            * self.consecutive_technical_miss
                                        ),
                                        self.max_technical_miss,
                                    )
                                    reward -= penalty
                                    self.consecutive_technical_miss += 1

                                    print(
                                        "PENALTY FROM CLOSING SIZE BEING TOO SHORT OR TOO BIG : ",
                                        penalty,
                                    )
                                    print("CLOSING SIZE : ", closing_size)

                            else:
                                # Entire position is closed

                                # Compute the realized pnl
                                realized_pnl = closing_size * (
                                    (effective_market_bid_price - self.positions[-1][1])
                                    / self.positions[-1][1]
                                )

                                # Update the balance
                                self.closing_fee = closing_size * self.market_fee
                                self.balance += (
                                    realized_pnl - self.closing_fee - self.opening_fee
                                ) + (closing_size / self.positions[-1][3])

                                self.realized_pnl = (
                                    realized_pnl - self.closing_fee - self.opening_fee
                                )

                                # Update the unrealized pnl
                                self.unrealized_pnl = 0

                                # Update the fees
                                self.closing_fee = 0
                                self.opening_fee = 0

                                # If the entire position is closed, remove it
                                self.positions = [
                                    p for p in self.positions if p[0] != "long"
                                ]  # Remove closed positions

                                self.consecutive_technical_miss = 0
                                self.consecutive_no_trade = 0

                                print(
                                    "ENTIRE POSITION IS CLOSED IMMEDIATELY, WEIGHT_DIFF : ",
                                    weight_diff,
                                )

                            print(
                                "POSITION SIZE BEFORE CLOSING SIZE : ",
                                current_position_size,
                            )
                            # position_size -= closing_size
                            print(
                                "POSITION SIZE AFTER CLOSING SIZE : ",
                                (current_position_size - closing_size),
                            )
                            print("CLOSING LONG SIZE : ", closing_size)

                        # Open a new short position with the remaining size
                        if (
                            current_position_size >= 0
                            and (abs(difference_size) - abs(current_position_size)) > 0
                            and desired_position_size < 0
                        ):
                            new_position_size = abs(difference_size) - abs(
                                current_position_size
                            )
                            print("DIFFERENCE SIZE NEW SHORT : ", abs(difference_size))
                            print(
                                "CURRENT POSITION SIZE NEW SHORT : ",
                                current_position_size,
                            )
                            print("NEW POSITION SIZE : ", new_position_size)
                            required_margin = new_position_size / self.leverage
                            print(
                                "REQUIRED MARGIN FOR THE NEW SHORT POSITION : ",
                                required_margin,
                            )
                            if self.balance >= (required_margin - epsilon):

                                # Calculate and deduct the transcation fee separately
                                self.opening_fee = new_position_size * self.market_fee

                                if self.limit_bounds:

                                    # Check if having a stop loss under the risk level is feasible
                                    max_loss_per_unit = (
                                        self.max_risk
                                        * (self.balance + self.position_value)
                                        - (2 * self.opening_fee)
                                    ) / new_position_size
                                    min_loss_per_unit = (
                                        self.min_risk
                                        * (self.balance + self.position_value)
                                        - (2 * self.opening_fee)
                                    ) / new_position_size

                                    # Restricted loss
                                    restricted_loss_per_unit = 1 - (
                                        (1 - self.cap_rate)
                                        * (mark_price / effective_market_bid_price)
                                    )

                                    # Compute the max
                                    final_max_loss_per_unit = min(
                                        max_loss_per_unit, restricted_loss_per_unit
                                    )

                                    print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                    print(
                                        "RESTRICTED LOSS PER UNIT : ",
                                        restricted_loss_per_unit,
                                    )
                                    print("MAX LOSS PER UNIT : ", max_loss_per_unit)
                                    print("MIN LOSS PER UNIT : ", min_loss_per_unit)
                                    adjusted_stop_loss = max(
                                        min(stop_loss, final_max_loss_per_unit),
                                        min_loss_per_unit,
                                    )
                                    print("ADJUSTED STOP LOSS : ", adjusted_stop_loss)

                                    if (
                                        adjusted_stop_loss > 0
                                        and adjusted_stop_loss >= min_loss_per_unit
                                    ):

                                        self.balance -= required_margin
                                        print(
                                            "BALANCE CHANGE (REQUIRED MARGIN NEW SHORT POSITION): ",
                                            -required_margin,
                                        )

                                        self.positions.append(
                                            (
                                                "short",
                                                effective_market_bid_price,
                                                new_position_size,
                                                self.leverage,
                                            )
                                        )

                                        # Update the unrealized pnl
                                        self.unrealized_pnl = new_position_size * (
                                            (effective_market_bid_price - rounded_ask)
                                            / effective_market_bid_price
                                        )
                                        self.unrealized_pnl -= self.opening_fee
                                        print(
                                            "ENTRY PRICE AND EFFECTIVE PRICE ARE THE SAME BECAUSE WE ARE IN A NEW SHORT POSITION. "
                                        )

                                        # Update the limit orders
                                        self.set_limits(
                                            self.positions[0][1],
                                            take_profit,
                                            "short",
                                            adjusted_stop_loss,
                                            mark_price,
                                        )

                                        # Update the cumuluative transactions costs
                                        self.cumulative_transaction_costs += (
                                            new_position_size * self.market_fee
                                        )

                                        # Update the metrics
                                        leverage_used = (
                                            new_position_size / max_position_size
                                        ) * self.leverage
                                        self.episode_metrics["leverage_used"].append(
                                            leverage_used
                                        )
                                        self.episode_metrics["num_trades"] += 1
                                        self.episode_metrics["list_trades"].append(
                                            self.episode_metrics["num_trades"]
                                        )

                                        self.consecutive_technical_miss = 0
                                        self.consecutive_no_trade = 0

                                        print("NEW POSITION CREATED FOR A NEW SHORT")

                                    else:
                                        # Put back the opening fees
                                        self.opening_fee -= (
                                            new_position_size * self.market_fee
                                        )

                                        # Penalize the model
                                        penalty = min(
                                            self.technical_miss
                                            + (
                                                0.5
                                                * self.technical_miss
                                                * self.consecutive_technical_miss
                                            ),
                                            self.max_technical_miss,
                                        )
                                        reward -= penalty
                                        self.consecutive_technical_miss += 1

                                        print("PENALTY FROM FEES : ", penalty)

                                        print(
                                            "WE DON'T OPEN THE NEW SHORT POSITION BECAUSE THE RISK TOLERANCE IS NOT FEASIBLE WITH THE FEES. "
                                        )

                                else:

                                    self.balance -= required_margin
                                    print(
                                        "BALANCE CHANGE (REQUIRED MARGIN NEW SHORT POSITION): ",
                                        -required_margin,
                                    )

                                    if self.positions:
                                        self.positions[-1] = (
                                            "short",
                                            effective_market_bid_price,
                                            new_position_size,
                                            self.leverage,
                                        )
                                    else:
                                        self.positions.append(
                                            (
                                                "short",
                                                effective_market_bid_price,
                                                new_position_size,
                                                self.leverage,
                                            )
                                        )

                                    # Update the unrealized pnl
                                    self.unrealized_pnl = new_position_size * (
                                        (effective_market_bid_price - rounded_ask)
                                        / effective_market_bid_price
                                    )
                                    self.unrealized_pnl -= self.opening_fee
                                    print(
                                        "ENTRY PRICE AND EFFECTIVE PRICE ARE THE SAME BECAUSE WE ARE IN A NEW SHORT POSITION. "
                                    )

                                    # Update the cumuluative transactions costs
                                    self.cumulative_transaction_costs += (
                                        new_position_size * self.market_fee
                                    )

                                    # Update the metrics
                                    leverage_used = (
                                        new_position_size / max_position_size
                                    ) * self.leverage
                                    self.episode_metrics["leverage_used"].append(
                                        leverage_used
                                    )
                                    self.episode_metrics["num_trades"] += 1
                                    self.episode_metrics["list_trades"].append(
                                        self.episode_metrics["num_trades"]
                                    )

                                    self.consecutive_technical_miss = 0
                                    self.consecutive_no_trade = 0

                                    print("NEW POSITION CREATED FOR A NEW SHORT")
                                    print("NO MORE RISK LIMIT")

                            else:
                                # Penalize the model
                                penalty = min(
                                    self.technical_miss
                                    + (
                                        0.5
                                        * self.technical_miss
                                        * self.consecutive_technical_miss
                                    ),
                                    self.max_technical_miss,
                                )
                                reward -= penalty
                                self.consecutive_technical_miss += 1

                                print("PENALTY FROM INITIAL MARGIN : ", penalty)

                                print(
                                    "NOT ENOUGH BALANCE TO OPEN THE NEW SHORT POSITION. CURRENT BALANCE : ",
                                    self.balance,
                                )

                    else:
                        # No changes in the weight

                        # Check if there is a position open first
                        # Check if despite not changing the position, the new stop loss is compatible with the risk tolerance
                        if self.positions:

                            (
                                position_type,
                                entry_price,
                                position_size,
                                previous_leverage,
                            ) = self.positions[-1]

                            # Update the unrealized pnl
                            self.unrealized_pnl = (
                                position_size
                                * ((rounded_bid - entry_price) / entry_price)
                                if position_type == "long"
                                else position_size
                                * ((entry_price - rounded_ask) / entry_price)
                            )
                            print("NO MORE RISK LIMIT")

                            if self.limit_bounds:

                                # Check if having a stop loss under the risk level is feasible
                                max_loss_per_unit = (
                                    self.max_risk * (self.balance + self.position_value)
                                    - (2 * self.opening_fee)
                                ) / position_size
                                min_loss_per_unit = (
                                    self.min_risk * (self.balance + self.position_value)
                                    - (2 * self.opening_fee)
                                ) / position_size

                                # Restricted loss
                                restricted_loss_per_unit = 1 - (
                                    (1 - self.cap_rate)
                                    * (mark_price / self.positions[-1][1])
                                )

                                # Compute the max
                                final_max_loss_per_unit = min(
                                    max_loss_per_unit, restricted_loss_per_unit
                                )

                                print("STOP LOSS WITH OUTPUT : ", stop_loss)
                                print(
                                    "RESTRICTED LOSS PER UNIT : ",
                                    restricted_loss_per_unit,
                                )
                                print("MAX LOSS PER UNIT : ", max_loss_per_unit)
                                print("MIN LOSS PER UNIT : ", min_loss_per_unit)
                                adjusted_stop_loss = max(
                                    min(stop_loss, final_max_loss_per_unit),
                                    min_loss_per_unit,
                                )
                                print("ADJUSTED STOP LOSS : ", adjusted_stop_loss)

                                if (
                                    adjusted_stop_loss > 0
                                    and adjusted_stop_loss >= min_loss_per_unit
                                ):

                                    self.set_limits(
                                        entry_price,
                                        take_profit,
                                        position_type,
                                        adjusted_stop_loss,
                                        mark_price,
                                    )

                                    # Update the unrealized pnl
                                    self.unrealized_pnl = (
                                        position_size
                                        * ((rounded_bid - entry_price) / entry_price)
                                        if position_type == "long"
                                        else position_size
                                        * ((entry_price - rounded_ask) / entry_price)
                                    )

                                else:

                                    # Execute the position type order to close the position
                                    realized_pnl = (
                                        position_size
                                        * (
                                            (effective_market_bid_price - entry_price)
                                            / entry_price
                                        )
                                        if position_type == "long"
                                        else position_size
                                        * (
                                            (entry_price - effective_market_ask_price)
                                            / entry_price
                                        )
                                    )

                                    # Calculate and deduct the transaction fee
                                    self.closing_fee = position_size * self.market_fee
                                    self.balance += (
                                        realized_pnl
                                        - self.closing_fee
                                        - self.opening_fee
                                    ) + (position_size / previous_leverage)

                                    self.realized_pnl = (
                                        realized_pnl
                                        - self.closing_fee
                                        - self.opening_fee
                                    )

                                    # Update the unrealized pnl
                                    self.unrealized_pnl = 0

                                    # Update the fees
                                    self.closing_fee = 0
                                    self.opening_fee = 0

                                    # Close the position if the stop loss can't meet the risk tolerance
                                    self.positions = []

                                    # Penalize the model
                                    reward -= 10

                                    print(
                                        "THE STOP LOSS DOESN'T MEET THE RISK TOLERANCE EVEN IF WE DIDN'T CHANGE WEIGHTS. "
                                    )

                            else:
                                # Penalize the model
                                penalty = min(
                                    self.no_trade
                                    + (0.5 * self.no_trade * self.consecutive_no_trade),
                                    self.max_no_trade,
                                )
                                reward -= penalty
                                self.consecutive_no_trade += 1

                                print("PENALTY FROM NO TRADE : ", penalty)

                                print(
                                    "NO POSITION OPEN DESPITE KEEPING THE SAME WEIGHT. "
                                )
                                print("NO MORE RISK LIMIT MODE")

                        else:
                            # Penalize the model
                            penalty = min(
                                self.no_trade
                                + (0.5 * self.no_trade * self.consecutive_no_trade),
                                self.max_no_trade,
                            )
                            reward -= penalty
                            self.consecutive_no_trade += 1

                            print("PENALTY FROM NO TRADE : ", penalty)

                            print("NO POSITION OPEN DESPITE KEEPING THE SAME WEIGHT. ")

                        print(
                            "NO CHANGES IN THE CURRENT POSITION. WE KEEP THE SAME WEIGHT. "
                        )

                else:
                    # Minimum size and trade amount aren't met

                    # Update the limit orders even if we keep the same position
                    # Check if there is an open position
                    if self.positions:

                        penalty = min(
                            self.technical_miss
                            + (
                                0.5
                                * self.technical_miss
                                * self.consecutive_technical_miss
                            ),
                            self.max_technical_miss,
                        )
                        reward -= penalty
                        self.consecutive_technical_miss += 1

                        print("PENALTY FROM NOT VALID SIZE : ", penalty)

                        if self.limit_bounds:

                            # Check if having a stop loss under the risk level is feasible
                            max_loss_per_unit = (
                                self.max_risk * (self.balance + self.position_value)
                                - (2 * self.opening_fee)
                            ) / self.positions[-1][2]
                            min_loss_per_unit = (
                                self.min_risk * (self.balance + self.position_value)
                                - (2 * self.opening_fee)
                            ) / self.positions[-1][2]

                            # Restricted loss
                            restricted_loss_per_unit = 1 - (
                                (1 - self.cap_rate)
                                * (mark_price / self.positions[-1][1])
                            )

                            # Compute the max
                            final_max_loss_per_unit = min(
                                max_loss_per_unit, restricted_loss_per_unit
                            )

                            print("STOP LOSS WITH OUTPUT : ", stop_loss)
                            print(
                                "RESTRICTED LOSS PER UNIT : ", restricted_loss_per_unit
                            )
                            print("MAX LOSS PER UNIT : ", max_loss_per_unit)
                            print("MIN LOSS PER UNIT : ", min_loss_per_unit)
                            adjusted_stop_loss = max(
                                min(stop_loss, final_max_loss_per_unit),
                                min_loss_per_unit,
                            )
                            print("ADJUSTED STOP LOSS : ", adjusted_stop_loss)

                            # Set new limits even if the position is the same
                            self.set_limits(
                                self.positions[-1][1],
                                take_profit,
                                self.positions[-1][0],
                                adjusted_stop_loss,
                                mark_price,
                            )

                            print(
                                "NEW LIMITS APPLIED EVEN IF THE MINIMUM AMOUNTS AREN'T FITTING. "
                            )

                        # Update the unrealized pnl
                        self.unrealized_pnl = (
                            self.positions[-1][2]
                            * (
                                (rounded_bid - self.positions[-1][1])
                                / self.positions[-1][1]
                            )
                            if self.positions[-1][0] == "long"
                            else self.positions[-1][2]
                            * (
                                (self.positions[-1][1] - rounded_ask)
                                / self.positions[-1][1]
                            )
                        )

                        print("DESIRED POSITION SIZE : ", desired_position_size)
                        print("DESIRED BTC SIZE : ", desired_btc_size)
                    else:
                        penalty = min(
                            self.technical_miss
                            + (
                                0.5
                                * self.technical_miss
                                * self.consecutive_technical_miss
                            ),
                            self.max_technical_miss,
                        )
                        reward -= penalty
                        self.consecutive_technical_miss += 1

                        print("PENALTY FROM NOT VALID SIZE : ", penalty)

                        print("NO POSITIONS OPEN AND THE DESIRED SIZE ISN'T ENOUGH. ")
                        print("DESIRED POSITION SIZE : ", desired_position_size)
                        print("DESIRED BTC SIZE : ", desired_btc_size)

        # Total value of positions
        if self.positions:
            self.position_value = (sum(p[2] for p in self.positions)) / self.positions[
                -1
            ][3]
            self.entry_price = self.positions[-1][1]
            if self.limit_bounds:
                self.stop_loss_price, _ = self.stop_loss_levels[self.entry_price]
                if self.take_profit_levels:
                    self.take_profit_price, _ = self.take_profit_levels[
                        self.entry_price
                    ]
                else:
                    self.take_profit_price = 0
        else:
            self.position_value = 0
            self.entry_price = 0
            if self.limit_bounds:
                self.stop_loss_price = 0
                self.take_profit_price = 0

        if self.limit_bounds:
            print("STOP LOSS LEVELS AT THE END OF STEP : ", self.stop_loss_levels)
        print(
            "CONSECUTIVE MISSED TECHNICAL ENTRIES : ", self.consecutive_technical_miss
        )
        print("CONSECUTIVE MISSED NO TRADE ENTRIES : ", self.consecutive_no_trade)

        # if self.positions:
        #     position_type = self.positions[-1][0]
        #     self.set_limits(
        #         self.positions[-1][1], stop_loss, take_profit, position_type
        #     )
        #     print("CURRENT POSITION : ", self.positions[-1])

        #     # Calculate the unrealized P&L
        #     if effective_price == self.positions[0][1]:
        #         print(
        #             "ENTRY PRICE AND EFFECTIVE PRICE ARE THE SAME. MEANING THAT THE POSITION IS NEW OR PARTIALLY CLOSED. "
        #         )
        #         self.unrealized_pnl = sum(
        #             (
        #                 p[2] * ((effective_price - p[1]) / p[1])
        #                 if p[0] == "long"
        #                 else p[2] * ((p[1] - effective_price) / p[1])
        #             )
        #             for p in self.positions
        #         )
        #     else:
        #         print(
        #             "ENTRY PRICE HASN'T BEEN MODIFIED, WE USE CURRENT PRICE FOR UNREALIZED PNL. "
        #         )
        #         self.unrealized_pnl = sum(
        #             (
        #                 p[2] * ((current_price - p[1]) / p[1])
        #                 if p[0] == "long"
        #                 else p[2] * ((p[1] - current_price) / p[1])
        #             )
        #             for p in self.positions
        #         )
        # else:
        #     self.unrealized_pnl = 0

        self.portfolio_value = self.balance + self.position_value
        self.portfolio_value = round(self.portfolio_value, 5)
        print("CURRENT PRICE : ", current_price)
        print("BALANCE : ", self.balance)
        print("TOTAL POSITIONS VALUE : ", self.position_value)
        print("UNREALIZED PNL : ", self.unrealized_pnl)
        print("PORTFOLIO VALUE : ", self.portfolio_value)
        print("POSITIONS : ", self.positions)
        self.history.append(self.portfolio_value)

        # Log metrics for the current step
        self.log_step_metrics()

        # Log trading return if no margin call was triggered
        if len(self.history) > 1:
            if not self.margin_call_triggered:
                if self.history[-2] == 0:
                    trading_return = 0
                else:
                    trading_return = (
                        self.history[-1] - self.history[-2]
                    ) / self.history[-2]
                self.trading_returns.append(trading_return)
                self.episode_metrics["returns"].append(trading_return)
                # Calculate log trading return
                if self.history[-1] <= 0 or self.history[-2] <= 0:
                    log_trading_return = 0
                else:
                    log_trading_return = np.log(self.history[-1] / self.history[-2])
                self.log_trading_returns.append(log_trading_return)
                final_return = (self.history[-1] - self.history[0]) / self.history[0]
                self.final_returns.append(final_return)
                reward += self.calculate_reward()
                self.episode_metrics["compounded_returns"].append(
                    self.compounded_returns - 1
                )
            else:
                reward += self.calculate_reward()
                self.episode_metrics["compounded_returns"].append(
                    self.compounded_returns - 1
                )
                self.margin_call_triggered = False  # Reset the flag for the next step
        else:
            reward += 0

        self.episode_metrics["rewards"].append(reward)
        self.metrics["rewards"].append(reward)

        # Compute the new margin price

        new_margin_price = self.calculate_liquidation_price()
        self.new_margin_price = new_margin_price
        print("NEXT MARGIN PRICE : ", self.new_margin_price)

        if self.limit_bounds:

            # Initialize the 11 additional variables + realized PnL
            additional_state = np.array(
                [
                    self.unrealized_pnl,
                    self.realized_pnl,
                    self.compounded_returns - 1,
                    self.take_profit_price,
                    self.stop_loss_price,
                    self.entry_price,
                    self.leverage,
                    self.allowed_leverage,
                    self.balance,
                    self.position_value,
                    self.desired_position_size,
                    self.current_position_size,
                    self.log_trading_returns[-1],
                    self.previous_max_dd,
                    self.closing_fee,
                    self.opening_fee,
                    self.current_ask,
                    self.current_bid,
                    self.mark_price,
                    self.margin_price,
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
                    self.new_margin_price,
                ]
            )

        else:
            additional_state = np.array(
                [
                    self.unrealized_pnl,
                    self.realized_pnl,
                    self.compounded_returns - 1,
                    self.entry_price,
                    self.leverage,
                    self.allowed_leverage,
                    self.balance,
                    self.position_value,
                    self.desired_position_size,
                    self.current_position_size,
                    self.log_trading_returns[-1],
                    self.previous_max_dd,
                    self.closing_fee,
                    self.opening_fee,
                    self.current_ask,
                    self.current_bid,
                    self.mark_price,
                    self.margin_price,
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
                    self.new_margin_price,
                ]
            )

        print("ADDITIONAL STATES : ")
        print()
        print("UNREALIZED PNL : ", self.unrealized_pnl)
        print("REALIZED PNL : ", self.realized_pnl)
        print("COMPOUNDED RETURNS : ", self.compounded_returns - 1)
        if self.limit_bounds:
            print("TAKE PROFIT PRICE : ", self.take_profit_price)
            print("STOP LOSS PRICE : ", self.stop_loss_price)
        print("ENTRY PRICE : ", self.entry_price)
        print("LEVERAGE : ", self.leverage)
        print("ALLOWED LEVERAGE ", self.allowed_leverage)
        print("BALANCE : ", self.balance)
        print("POSITION VALUE : ", self.position_value)
        print("DESIRED POSITION SIZE : ", self.desired_position_size)
        print("LOG RETURN : ", self.log_trading_returns[-1])
        print("PREVIOUS MAX DRAWDOWN : ", self.previous_max_dd)
        print("CLOSING FEES : ", self.closing_fee)
        print("OPENING FEES : ", self.opening_fee)
        print("CURRENT ASK : ", self.current_ask)
        print("CURRENT BID : ", self.current_bid)
        print("MARK PRICE : ", self.mark_price)

        # terminated = self.current_step >= (self.start_idx + self.episode_length)

        self.current_step += 1
        next_state, terminated, info = self.update_sequences(additional_state)
        self.state = next_state

        # next_state_numpy, terminated_numpy, info_numpy = self.update_sequences_numpy(
        #     additional_state
        # )
        # self.state_numpy = next_state_numpy

        # Reset the episode if not enough value in the portfolio

        if self.portfolio_value < 0.1 * self.initial_balance:
            print(
                "EPISODE TERMINATED EARLIER DUE TO PORTFOLIO NOT HAVING ENOUGH VALUE. "
            )
            reward -= 100
            terminated = True

        if abs(reward) > 10000000:
            terminated = True

        if terminated and abs(reward) <= 10000000:
            print("EPISODE TERMINATED")
            print("CURRENT STEP : ", self.current_step)
            print("START INDEX : ", self.start_idx)
            print("END INDEX : ", self.end_idx)
            # next_state, info = self.data[self.current_step - 1], {}
            # self.state = next_state
            self.log_episode_metrics()
            self.metrics["compounded_returns"].append(
                self.episode_metrics["compounded_returns"][-1]
            )

            if self.current_episode < self.total_episodes:
                terminated = False
                self.reset()
            else:
                avg_compounded_returns = np.mean(self.metrics["compounded_returns"])
                print(
                    f"AVERAGE COMPOUNDED RETURNS OVER {self.total_episodes} EPISODES : {avg_compounded_returns}"
                )
                total_rewards = np.sum(self.metrics["rewards"])
                mean_rewards = np.mean(self.metrics["rewards"])
                std_rewards = np.std(self.metrics["rewards"])
                skew_rewards = calculate_skewness(self.metrics["rewards"])
                kurt_rewards = calculate_kurtosis(self.metrics["rewards"])
                print(f"STATS ON REWARDS OVER {self.total_episodes} EPISODES : ")
                print()
                print("TOTAL REWARDS : ", total_rewards)
                print("MEAN REWARDS : ", mean_rewards)
                print("STD REWARDS : ", std_rewards)
                print("SKEWNESS REWARDS : ", skew_rewards)
                print("KURTOSIS REWARDS : ", kurt_rewards)
                print()
                print()
                if len(self.metrics["returns"]) != 0:
                    total_returns = np.sum(self.metrics["returns"])
                    mean_returns = np.mean(self.metrics["returns"])
                    std_returns = np.std(self.metrics["returns"])
                    skew_returns = calculate_skewness(self.metrics["returns"])
                    kurt_returns = calculate_kurtosis(self.metrics["returns"])
                    print(f"STATS ON RETURNS OVER {self.total_episodes} EPISODES : ")
                    print()
                    print("TOTAL RETURNS : ", total_returns)
                    print("MEAN RETURNS : ", mean_returns)
                    print("STD RETURNS : ", std_returns)
                    print("SKEWNESS RETURNS : ", skew_returns)
                    print("KURTOSIS RETURNS : ", kurt_returns)
                print(
                    "METRICS FINAL COMPOUNDED RETURN : ",
                    self.metrics["compounded_returns"],
                )

        # else:
        #     next_state = self.data[self.current_step]
        #     self.state = next_state
        #     info = {}

        truncated = False

        print("REWARD END OF STEP : ", reward)

        return next_state, reward, terminated, truncated, info

    def set_limits(
        self, current_price, take_profit, position_type, adjusted_stop_loss, mark_price
    ):

        if not self.positions:
            return

        most_recent_position = self.positions[-1]
        entry_price = most_recent_position[1]

        # # Calculate the total position size
        # total_position_size = sum([pos[2] for pos in self.positions])

        # max_loss_per_unit = (
        #     0.02 * (self.balance + self.position_value) - (2 * self.opening_fee)
        # ) / total_position_size
        # min_loss_per_unit = (
        #     0.001 * (self.balance + self.position_value) - (2 * self.opening_fee)
        # ) / total_position_size

        # print("STOP LOSS WITH OUTPUT : ", stop_loss)
        # print("MAX LOSS PER UNIT : ", max_loss_per_unit)
        # print("MIN LOSS PER UNIT : ", min_loss_per_unit)
        # adjusted_stop_loss = max(min(stop_loss, max_loss_per_unit), min_loss_per_unit)
        # print("ADJUSTED STOP LOSS : ", adjusted_stop_loss)
        stop_loss_price = (
            current_price * (1 - adjusted_stop_loss)
            if position_type == "long"
            else current_price * (1 + adjusted_stop_loss)
        )
        # Round the price limit
        stop_loss_price = (
            round(stop_loss_price / self.price_precision) * self.price_precision
        )

        # Cap/Floor limit rate for the take profit
        restricted_profit = 1 - ((1 - self.cap_rate) * (mark_price / entry_price))

        # Minimum take profit to set
        min_profit_per_unit = (2 * self.opening_fee + self.min_profit) / self.positions[
            -1
        ][2]
        adjusted_tp = max(min_profit_per_unit, take_profit)

        print("ADJUSTED TAKE PROFIT : ", adjusted_tp)
        print("MIN PROFIT PER UNIT : ", min_profit_per_unit)
        print("2 TIMES OPENING FEES : ", 2 * self.opening_fee)

        if take_profit <= restricted_profit:
            take_profit_price = (
                current_price * (1 + adjusted_tp)
                if position_type == "long"
                else current_price * (1 - adjusted_tp)
            )
            # Round the price limit
            take_profit_price = (
                round(take_profit_price / self.price_precision) * self.price_precision
            )

            print("STOP LOSS PRICE : ", stop_loss_price)
            print("TAKE PROFIT PRICE : ", take_profit_price)

            self.stop_loss_levels[entry_price] = (stop_loss_price, position_type)
            self.take_profit_levels[entry_price] = (take_profit_price, position_type)
        else:
            # Only set stop-loss without take-profit
            print("STOP LOSS PRICE (NO TAKE PROFIT SET): ", stop_loss_price)
            self.stop_loss_levels[entry_price] = (stop_loss_price, position_type)
            # Optionally, you can still log that take-profit wasn't set
            print("TAKE PROFIT NOT SET DUE TO EXCEEDING CAP/FLOOR LIMITS. ")

    def check_limits(self, high_price, low_price):

        if not self.positions:
            print("NO POSITIONS IN THE CHECK_LIMITS METHOD. ")
            return

        # # Calculate the total position value based on worst-case scenario
        # worst_case_price = (
        #     low_price if any(p[0] == "long" for p in self.positions) else high_price
        # )

        # if self.portfolio_value < (self.maintenance_margin_rate * self.initial_balance):
        #     print("MARGIN CALL BECAUSE NOT ENOUGH ON PORTFOLIO VALUE. ")
        #     print("PORTFOLIO BEFORE THE RESET : ", self.portfolio_value)
        #     self.handle_margin_call(worst_case_price)
        #     return

        most_recent_position = self.positions[-1]  # Get the most recent position
        entry_price = most_recent_position[1]

        if entry_price not in self.stop_loss_levels:
            print(f"Stop-loss not set for entry price: {entry_price}")
            return

        stop_loss_price, position_type = self.stop_loss_levels[entry_price]
        take_profit_price = None

        if entry_price in self.take_profit_levels:
            take_profit_price, _ = self.take_profit_levels[entry_price]

        if position_type == "long":
            print("THIS IS A LONG POSITION. ")
            print("LOW PRICE OF THE LONG POSITION : ", low_price)
            print("HIGH PRICE OF THE LONG POSITION : ", high_price)
            if low_price <= stop_loss_price and (
                take_profit_price is not None and high_price >= take_profit_price
            ):
                execution_price = stop_loss_price
                self.episode_metrics["stop_loss_hits"] += 1  # Count stop-loss hit
                self.execute_order("sell", execution_price, entry_price)
                del self.stop_loss_levels[entry_price]
                del self.take_profit_levels[entry_price]
                print("STOP LOSS PRICE EXECUTED FOR A WORST CASE ESTIMATION IN LONG. ")

            elif low_price <= stop_loss_price or (
                take_profit_price is not None and high_price >= take_profit_price
            ):
                execution_price = (
                    stop_loss_price
                    if low_price <= stop_loss_price
                    else take_profit_price
                )
                if take_profit_price is not None and high_price >= take_profit_price:
                    print("TAKE PROFIT HIT FOR THE LONG POSITION")
                    print(
                        "HIGH PRICE AND TAKE PROFIT PRICE FOR LONG : ",
                        high_price,
                        take_profit_price,
                    )
                if low_price <= stop_loss_price:
                    print("STOP LOSS HIT FOR THE LONG POSITION")
                    print(
                        "LOW PRICE AND STOP LOSS PRICE FOR LONG : ",
                        low_price,
                        stop_loss_price,
                    )
                    self.episode_metrics["stop_loss_hits"] += 1  # Count stop-loss hit
                self.execute_order("sell", execution_price, entry_price)
                del self.stop_loss_levels[entry_price]
                if take_profit_price is not None:
                    del self.take_profit_levels[entry_price]

            else:
                del self.stop_loss_levels[entry_price]
                if take_profit_price is not None:
                    del self.take_profit_levels[entry_price]
                print("THE LIMIT ORDERS WEREN'T TRIGGERED FOR THIS OBSERVATION. ")

        elif position_type == "short":
            print("THIS IS A SHORT POSITION. ")
            print("LOW PRICE OF THE SHORT POSITION : ", low_price)
            print("HIGH PRICE OF THE SHORT POSITION : ", high_price)
            if high_price >= stop_loss_price and (
                take_profit_price is not None and low_price <= take_profit_price
            ):
                execution_price = stop_loss_price
                self.episode_metrics["stop_loss_hits"] += 1  # Count stop-loss hit
                self.execute_order("buy", execution_price, entry_price)
                del self.stop_loss_levels[entry_price]
                del self.take_profit_levels[entry_price]
                print("STOP LOSS PRICE EXECTUED FOR A WORST CASE ESTIMATION IN SHORT. ")

            elif high_price >= stop_loss_price or (
                take_profit_price is not None and low_price <= take_profit_price
            ):
                execution_price = (
                    stop_loss_price
                    if high_price >= stop_loss_price
                    else take_profit_price
                )
                if take_profit_price is not None and low_price <= take_profit_price:
                    print("TAKE PROFIT HIT FOR THE SHORT POSITION")
                    print(
                        "LOW PRICE AND TAKE PROFIT PRICE FOR SHORT : ",
                        low_price,
                        take_profit_price,
                    )
                if high_price >= stop_loss_price:
                    print("STOP LOSS HIT FOR THE SHORT POSITION")
                    print(
                        "HIGH PRICE AND STOP LOSS PRICE FOR SHORT : ",
                        high_price,
                        stop_loss_price,
                    )
                    self.episode_metrics["stop_loss_hits"] += 1  # Count stop-loss hit
                self.execute_order("buy", execution_price, entry_price)
                del self.stop_loss_levels[entry_price]
                if take_profit_price is not None:
                    del self.take_profit_levels[entry_price]

            else:
                del self.stop_loss_levels[entry_price]
                if take_profit_price is not None:
                    del self.take_profit_levels[entry_price]
                print("THE LIMIT ORDERS WEREN'T TRIGGERED FOR THIS OBSERVATION. ")

    def execute_order(
        self, order_type, execution_price, entry_price, margin_price=None
    ):

        # Apply slippage to the execution price
        # slippage = np.random.normal(self.slippage_mean, self.slippage_std)
        # execution_price *= 1 + slippage if order_type == "buy" else 1 - slippage

        if not self.positions:
            print("NO POSITIONS (EXECUTE_ORDER)")
            return

        print("POSITIONS BEFORE POP : ", self.positions)
        position = self.positions.pop(0)
        print("POSITIONS AFTER POP : ", self.positions)
        position_type, entry_price, position_size, previous_leverage = position

        print(
            "DIFFERENCE EFF PRICE AND CURR PRICE IN EXECUTE_ORDER : ",
            (execution_price - entry_price),
        )

        # if position_type == "long" and order_type == "sell":
        #    profit_loss = position_size * (execution_price - entry_price)
        #    self.balance += profit_loss * (1 - self.market_fee)
        #    if not self.margin_call_triggered:
        #        print("PROFIT FROM SHORT LIMIT ORDER : ", profit_loss)
        #    else:
        #        print("LOSS FROM MARGIN CALL LIQUIDATION : ", profit_loss)
        # elif position_type == "short" and order_type == "buy":
        #    profit_loss = position_size * (entry_price - execution_price)
        #    self.balance += profit_loss * (1 - self.market_fee)
        #    if not self.margin_call_triggered:
        #        print("PROFIT FROM SHORT LIMIT ORDER : ", profit_loss)
        #    else:
        #        print("LOSS FROM MARGIN CALL LIQUIDATION : ", profit_loss)

        if not self.margin_call_triggered:
            if position_type == "long" and order_type == "sell":

                # Compute the realized profit
                profit_loss = position_size * (
                    (execution_price - entry_price) / entry_price
                )
                print("EXECUTION PRICE : ", execution_price)
                print("ENTRY PRICE : ", entry_price)
                print("BALANCE BEFORE LONG LIMIT ORDER : ", self.balance)

                # Update the balance
                self.closing_fee = position_size * self.limit_fee
                self.balance += (profit_loss - self.closing_fee - self.opening_fee) + (
                    position_size / previous_leverage
                )
                self.realized_pnl = profit_loss - self.closing_fee - self.opening_fee

                print(
                    "PROFIT FROM LONG LIMIT ORDER : ",
                    (profit_loss - self.closing_fee - self.opening_fee),
                )
                print("BALANCE AFTER LONG LIMIT ORDER EXECUTED : ", self.balance)

            elif position_type == "short" and order_type == "buy":

                # Compute the realized pnl
                profit_loss = position_size * (
                    (entry_price - execution_price) / entry_price
                )
                print("EXECUTION PRICE : ", execution_price)
                print("ENTRY PRICE : ", entry_price)
                print("BALANCE BEFORE SHORT LIMIT ORDER : ", self.balance)

                # Update the balance
                self.closing_fee = position_size * self.limit_fee
                self.balance += (profit_loss - self.closing_fee - self.opening_fee) + (
                    position_size / previous_leverage
                )
                self.realized_pnl = profit_loss - self.closing_fee - self.opening_fee

                print(
                    "PROFIT FROM SHORT LIMIT ORDER : ",
                    (profit_loss - self.closing_fee - self.opening_fee),
                )
                print("BALANCE AFTER SHORT LIMIT ORDER EXECUTED : ", self.balance)

        else:
            # Adapt the order if it is coming from a margin call

            if position_type == "long" and order_type == "sell":

                # Compute the loss
                realized_profit = position_size * (
                    (margin_price - entry_price) / entry_price
                )
                print("MARGIN PRICE USED TO CLOSE THE POSITION : ", margin_price)

                # Update the balance
                self.closing_fee = position_size * self.limit_fee
                margin_fee = position_size * self.liquidation_fee
                self.margin_fee = margin_fee
                # self.balance += (
                #     realized_profit - self.closing_fee - self.opening_fee - margin_fee
                # ) + (position_size / previous_leverage)

                self.realized_pnl = (
                    realized_profit - self.closing_fee - self.opening_fee - margin_fee
                )

                if (
                    self.balance
                    + self.realized_pnl
                    + (position_size / previous_leverage)
                    < 0
                ):
                    self.realized_pnl = -(
                        self.balance + (position_size / previous_leverage)
                    )

                self.balance = (
                    self.balance
                    + self.realized_pnl
                    + (position_size / previous_leverage)
                )

                print("OPENING FEES : ", self.opening_fee)

                print(
                    "LOSS FROM MARGIN CALL LIQUIDATION OF THE LONG POSITION : ",
                    (self.realized_pnl),
                )
                print(
                    "BALANCE AFTER MARGIN CALL OF THE LONG POSITION EXECUTED : ",
                    self.balance,
                )

            elif position_type == "short" and order_type == "buy":

                # Compute the loss
                realized_profit = position_size * (
                    (entry_price - margin_price) / entry_price
                )
                print("MARGIN PRICE USED TO CLOSE THE POSITION : ", margin_price)

                # Update the balance
                self.closing_fee = position_size * self.limit_fee
                margin_fee = position_size * self.liquidation_fee
                self.margin_fee = margin_fee
                # self.balance += (
                #     realized_profit - self.closing_fee - self.opening_fee - margin_fee
                # ) + (position_size / previous_leverage)

                self.realized_pnl = (
                    realized_profit - self.closing_fee - self.opening_fee - margin_fee
                )

                if (
                    self.balance
                    + self.realized_pnl
                    + (position_size / previous_leverage)
                    < 0
                ):
                    self.realized_pnl = -(
                        self.balance + (position_size / previous_leverage)
                    )

                self.balance = (
                    self.balance
                    + self.realized_pnl
                    + (position_size / previous_leverage)
                )

                print("OPENING FEES : ", self.opening_fee)

                print(
                    "LOSS FROM MARGIN CALL LIQUIDATION OF THE SHORT POSITION : ",
                    (self.realized_pnl),
                )
                print(
                    "BALANCE AFTER MARGIN CALL OF THE SHORT POSITION EXECUTED : ",
                    self.balance,
                )

        # Ensure the unrealized pnl is closed
        self.unrealized_pnl = 0

        # Ensure the initial margin is 0
        self.position_value = 0

        # Ensure all positions are closed
        self.positions = []

        # Update portfolio value after closing the position
        self.portfolio_value = self.balance

        # Reset the fees
        self.closing_fee = 0
        self.opening_fee = 0

    def calculate_liquidation_price(self):

        if not self.positions:
            margin_price = 0
            return margin_price

        position_type, entry_price, position_size, previous_leverage = self.positions[
            -1
        ]

        # Get dynamic leverage and margin rate based on notional value
        leverage, maintenance_margin_rate, maintenance_amount, _, _ = (
            self.get_margin_tier(position_size)
        )

        # Calculate the required margin using the dynamic rate
        required_margin = maintenance_margin_rate * position_size - maintenance_amount

        # Calculate the margin call price
        margin_price = (
            (entry_price * (-(self.portfolio_value - required_margin) / position_size))
            + entry_price
            if position_type == "long"
            else entry_price
            - (
                entry_price
                * (-(self.portfolio_value - required_margin) / position_size)
            )
        )

        margin_price = max(margin_price, 0)

        return margin_price

    def check_margin_call(self, high_price, low_price):

        if not self.positions:
            self.margin_price = 0
            print("NO POSITIONS IN THE CHECK_MARGIN_CALL METHOD. ")
            return

        # Calculate the total position value based on worst-case scenario
        worst_case_price = (
            low_price if any(p[0] == "long" for p in self.positions) else high_price
        )

        # worst_case_size = sum(
        #     p[2] * (worst_case_price if p[0] == "long" else p[1])
        #     for p in self.positions
        # )

        position_type, entry_price, position_size, previous_leverage = self.positions[
            -1
        ]

        # Get dynamic leverage and margin rate based on notional value
        leverage, maintenance_margin_rate, maintenance_amount, _, _ = (
            self.get_margin_tier(position_size)
        )

        # Calculate the required margin using the dynamic rate
        required_margin = maintenance_margin_rate * position_size - maintenance_amount

        # Calculate the margin call price
        margin_price = (
            (entry_price * (-(self.portfolio_value - required_margin) / position_size))
            + entry_price
            if position_type == "long"
            else entry_price
            - (
                entry_price
                * (-(self.portfolio_value - required_margin) / position_size)
            )
        )

        margin_price = max(margin_price, 0)
        self.margin_price = margin_price

        if self.limit_bounds:
            stop_loss_price, position_type = self.stop_loss_levels[entry_price]

        print("MARGIN PRICE LIMIT : ", margin_price)
        if self.limit_bounds:
            print("STOP LOSS PRICE : ", self.stop_loss_levels[entry_price])
        print("WORST CASE PRICE : ", worst_case_price)

        # Check if a margin call is triggered
        # if self.portfolio_value < self.refill_rate * self.initial_balance:
        #     self.handle_margin_call(worst_case_price, margin_price)
        #     print("NOT ENOUGH BALANCE TO CONTINUE THE TRADING, REFILLING THE ACCOUNT. ")

        if self.limit_bounds:

            if (
                position_type == "long"
                and margin_price > worst_case_price
                and margin_price > stop_loss_price
            ):
                self.handle_margin_call(worst_case_price, margin_price)
                print("MARGIN CALL FOR LONG POSITION TRIGGERED. ")

            elif (
                position_type == "short"
                and margin_price < worst_case_price
                and margin_price < stop_loss_price
            ):
                self.handle_margin_call(worst_case_price, margin_price)
                print("MARGIN CALL FOR SHORT POSITION TRIGGERED. ")

            else:
                print("NO MARGIN CALL TRIGGERED. ")

        else:

            if position_type == "long" and margin_price > worst_case_price:
                self.handle_margin_call(worst_case_price, margin_price)
                print("MARGIN CALL FOR LONG POSITION TRIGGERED. ")

            elif position_type == "short" and margin_price < worst_case_price:
                self.handle_margin_call(worst_case_price, margin_price)
                print("MARGIN CALL FOR SHORT POSITION TRIGGERED. ")

            else:
                print("NO MARGIN CALL TRIGGERED. ")

            print("NO MORE RISK LIMIT MODE")

        # TO DO : MODIFY THIS
        # elif self.portfolio_value < (
        #     self.maintenance_margin_rate * self.initial_balance
        # ) or self.portfolio_value < (
        #     self.maintenance_margin_rate * self.portfolio_value
        # ):
        #     self.handle_margin_call(worst_case_price)
        #     print("MARGIN CALL BECAUSE NOT ENOUGH VALUE IN THE PORTFOLIO. ")

        # # If the balance is less than the required margin, trigger a margin call. 5% of the balance must remain.
        # if available_margin < required_maintenance_margin:
        #     print(
        #         "MARGIN CALL TRIGGERED, AVAILABLE MARGIN IS UNDER THE REQUIRED MARGIN : ",
        #         required_maintenance_margin,
        #     )
        #     print("WORST CASE PRICE AT WHICH THE CHECK WAS MADE : ", worst_case_price)
        #     self.handle_margin_call(worst_case_price)

    def handle_margin_call(self, worst_case_price, margin_price):

        # Mark that a margin call was triggered
        self.margin_call_triggered = True

        if not self.positions:
            print("NO POSITIONS (HANDLE_MARGIN_CALL)")
            return

        # Liquidate all positions to cover the margin call
        for position in self.positions:
            position_type, entry_price, position_size, previous_leverage = position
            order_type = "sell" if position_type == "long" else "buy"
            self.execute_order(
                order_type, worst_case_price, entry_price, margin_price=margin_price
            )

        if self.limit_bounds:
            # Clear all stop loss and take profit levels
            self.stop_loss_levels.clear()
            self.take_profit_levels.clear()

        self.episode_metrics["num_margin_calls"] += 1  # Log the number of margin calls
        self.episode_metrics["list_margin_calls"].append(
            self.episode_metrics["num_margin_calls"]
        )

        # Log the trading return before resetting, without counting the reset as a return
        if len(self.history) > 0:
            if self.history[-1] == 0:
                trading_return = 0
            else:
                trading_return = (
                    self.portfolio_value - self.history[-1]
                ) / self.history[-1]
            print("HISTORY[-1] : ", self.history[-1])
            print("PORTFOLIO VALUE AT THE HANDLE MARGIN CALL : ", self.portfolio_value)
            self.trading_returns.append(trading_return)
            self.episode_metrics["returns"].append(trading_return)
            # Calculate log trading return
            if self.portfolio_value <= 0 or self.history[-1] <= 0:
                log_trading_return = 0
            else:
                log_trading_return = np.log(self.portfolio_value / self.history[-1])
            self.log_trading_returns.append(log_trading_return)
            final_return = (self.portfolio_value - self.history[0]) / self.history[0]
            self.final_returns.append(final_return)

        # Reset balance to initial value
        # print("BALANCE BEFORE MARGIN RESET : ", self.balance)
        # if self.portfolio_value < self.initial_balance * self.refill_rate:
        #     self.balance = self.initial_balance
        #     self.portfolio_value = self.balance
        # self.balance = self.initial_balance
        # self.portfolio_value = self.balance
        print("BALANCE AFTER MARGIN RESET : ", self.balance)

    def get_margin_tier(self, notional_value):
        # Example tiered structure for BTCUSDT, adjust according to Binance's rules
        if 0 <= notional_value <= 50000:
            return (
                125,
                0.004,
                0,
                0,
                50000,
            )  # Leverage, Maintenance Margin Rate, Maintenance amount (USDT), lower position size limit, higher position size limit for Tier 1
        elif 50000 < notional_value <= 600000:
            return (
                100,
                0.005,
                50,
                50000,
                600000,
            )  # Leverage, Maintenance Margin Rate, Maintenance amount (USDT) for Tier 2
        elif 600000 < notional_value <= 3000000:
            return 75, 0.0065, 950, 600000, 3000000
        elif 3000000 < notional_value <= 12000000:
            return 50, 0.01, 11450, 3000000, 12000000
        elif 12000000 < notional_value <= 70000000:
            return 25, 0.02, 131450, 12000000, 70000000
        elif 70000000 < notional_value <= 100000000:
            return 20, 0.025, 481450, 70000000, 100000000
        elif 100000000 < notional_value <= 230000000:
            return 10, 0.05, 2981450, 100000000, 230000000
        elif 230000000 < notional_value <= 480000000:
            return 5, 0.1, 14481450, 230000000, 480000000
        elif 480000000 < notional_value <= 600000000:
            return 4, 0.125, 26481450, 480000000, 600000000
        elif 600000000 < notional_value <= 800000000:
            return 3, 0.15, 41481450, 600000000, 800000000
        elif 800000000 < notional_value <= 1200000000:
            return 2, 0.25, 121481450, 800000000, 1200000000
        elif 1200000000 < notional_value <= 1800000000:
            return 1, 0.5, 421481450, 1200000000, 1800000000

    def calculate_reward(self):
        if len(self.history) < 2:
            return 0  # Not enough data to calculate reward

        returns = np.array(self.trading_returns)
        if len(returns) < 1 or np.isnan(returns).any():
            return 0  # Not enough data to calculate returns
        print("RETURNS : ", returns[-1])
        self.metrics["returns"].append(returns[-1])

        # log_returns_factor = 1
        # compounded_returns_factor = 1

        # if returns[-1] > 0:
        #     log_returns_factor = 1.1
        # if returns[-1] >= self.max_risk:
        #     log_returns_factor = 1.3

        # var = calculate_empirical_var(returns, 0.99)
        # print("VALUE AT RISK : ", var)
        # es = calculate_empirical_es(returns, 0.99)
        # print("EXPECTED SHORTFALL : ", es)
        max_dd = calculate_max_drawdown(self.history)
        print("MAX DD : ", max_dd)

        # Rolling sharpe ratio

        WINDOW_SIZE = min(len(returns), 30)
        RISK_FREE_RATE = (0.005 / (365.25 * 24 * 60)) * 5
        ROLLING_SHARPE_FACTOR = WINDOW_SIZE / 30

        window_returns = returns[-WINDOW_SIZE:]
        mean_return = np.mean(window_returns)
        std_return = np.std(window_returns)
        rolling_sharpe = (
            (mean_return - RISK_FREE_RATE) / std_return if std_return != 0 else 0
        )
        print("MEAN RETURN ROLLING SHARPE : ", mean_return)
        print("STD ROLLING SHARPE : ", std_return)

        self.episode_metrics["rolling_sharpe"].append(rolling_sharpe)
        self.sharpe_ratio = rolling_sharpe

        # Add log-returns as reward
        log_returns = np.array(self.log_trading_returns)
        if len(log_returns) < 1 or np.isnan(log_returns).any():
            return 0  # Not enough data to calculate returns
        self.episode_metrics["log_returns"].append(log_returns[-1])
        self.metrics["log_returns"].append(log_returns[-1])

        # Step adjusted risk-log_returns

        if self.positions:
            position_type, entry_price, position_size, leverage = self.positions[-1]
            current_risk = (
                (
                    (abs(entry_price - self.margin_price) / entry_price)
                    + self.liquidation_fee
                )
                * position_size
                + 2 * self.opening_fee
            ) / self.portfolio_value
            print("CURRENT RISK BEFORE ADJUSTMENT : ", current_risk)
            current_risk = min(current_risk, 1)
            print("ENTRY PRICE USED FOR CURRENT RISK : ", entry_price)
            print("POSITION SIZE USED FOR CURRENT RISK : ", position_size)
            print("MARGIN PRICE USED FOR CURRENT RISK : ", self.margin_price)
            print("OPENING FEES USED FOR CURRENT RISK : ", self.opening_fee)
            print("PORTFOLIO VALUE USED FOR CURRENT RISK : ", self.portfolio_value)
        else:
            current_risk = 0

        self.episode_metrics["step_risk"].append(current_risk)
        self.current_risk = current_risk

        # Adjust step return by current risk
        if current_risk > 0:
            risk_adjusted_step_return = log_returns[-1] / current_risk
        else:
            risk_adjusted_step_return = log_returns[-1]

        self.episode_metrics["risk_adjusted_step_return"].append(
            risk_adjusted_step_return
        )
        self.risk_adjusted_step = risk_adjusted_step_return

        std_dev = np.std(returns) if len(returns) > 1 else 0
        # print("STD_DEV : ", std_dev)
        # skewness = calculate_skewness(returns)
        # print("SKEWNESS : ", skewness)
        # kurtosis = calculate_kurtosis(returns)
        # print("KURTOSIS : ", kurtosis)
        sharpe_ratio = np.mean(returns) / std_dev if std_dev != 0 else 0
        self.episode_metrics["sharpe_ratio"].append(sharpe_ratio)
        self.metrics["sharpe_ratio"].append(sharpe_ratio)
        # print("SHARPE_RATIO : ", sharpe_ratio)

        penalty = 0

        if max_dd > self.previous_max_dd:
            # Penalize for increased drawdown
            penalty -= max_dd

        self.previous_max_dd = max_dd

        # stop_loss_total_risk = sum(
        #    [self.stop_loss_levels[price][0] for price in self.stop_loss_levels]
        # )
        # if stop_loss_total_risk > 0.02 * self.balance:
        #    penalty -= 100

        # Check if there were margin calls and apply penalty to the reward
        if self.margin_call_triggered:
            margin_call_penalty = 0
        else:
            margin_call_penalty = 0

        # Add penalty for the number of stop-loss hits
        stop_loss_hits = self.episode_metrics.get("stop_loss_hits", 0)
        penalty -= stop_loss_hits * 0

        profit_loss_ratio = 0

        if returns[-1] > 0:
            self.profits.append(returns[-1])
        elif returns[-1] < 0:
            self.losses.append(returns[-1])

        if len(self.profits) == 0 or (len(self.profits) == 0 and len(self.losses) == 0):
            profit_loss_ratio = 0
        elif len(self.losses) == 0 and len(self.profits) != 0:
            profit_loss_ratio = np.mean(self.profits)
        elif len(self.profits) != 0 and len(self.losses) != 0:
            average_losses = np.mean(self.losses)
            average_losses = abs(average_losses)
            if average_losses != 0:
                profit_loss_ratio = np.mean(self.profits) / average_losses
                print("AVERAGE PROFITS : ", np.mean(self.profits))
            else:
                profit_loss_ratio = np.mean(self.profits)

        if profit_loss_ratio < 1:
            profit_loss_ratio = 0

        profit_risk_ratio = returns[-1] / self.max_risk
        profit_risk_factor = 1
        if profit_risk_ratio >= 1:
            profit_risk_factor = 1.5
        self.episode_metrics["profit_risk_ratio"].append(profit_risk_ratio)
        self.metrics["profit_risk_ratio"].append(profit_risk_ratio)

        # # Add final returns as reward
        # final_return = np.array(self.final_returns)
        # if len(final_return) < 1 or np.isnan(final_return).any():
        #     return 0  # Not enough data to calculate returns

        # Initialize compounded return
        self.compounded_returns *= 1 + returns[-1]

        # Calculate final compounded return
        final_compounded_return = self.compounded_returns - 1

        # if final_compounded_return > 0:
        #     compounded_returns_factor = 1.1

        # Activity reward
        activity_reward = 0
        if self.positions:
            activity_reward = 0.001

        # Only at the end of episode compounded return
        end_episode_comp_return = 0
        if (self.current_step + 1) >= (self.start_idx + self.episode_length):
            print("CURRENT STEP FROM REWARD : ", self.current_step)
            print(
                "START + EPISODE LENGTH FROM REWARD : ",
                (self.start_idx + self.episode_length),
            )
            end_episode_comp_return = final_compounded_return

        # Reward leverage if returns are positive
        leverage_factor = 0.001 * self.previous_leverage
        leverage_bonus = 0
        if log_returns[-1] > leverage_factor:
            leverage_bonus = self.previous_leverage * 0.01

        # Reward compounded returns if it is positive
        compounded_return_bonus = 0
        if final_compounded_return > 0:
            compounded_return_bonus = final_compounded_return

        reward = (
            # log_returns[-1]
            risk_adjusted_step_return * 0.5
            + rolling_sharpe * 0.5 * ROLLING_SHARPE_FACTOR
            # final_compounded_return
            # + penalty
            # + activity_reward
            + end_episode_comp_return * 10
            + leverage_bonus
            # + compounded_return_bonus
        )

        print("LEVERAGE BONUS : ", leverage_bonus)
        print("END OF EPISODE COMPOUNDED RETURNS : ", end_episode_comp_return)
        print("RISK ADJUSTED STEP LOG-RETURN : ", risk_adjusted_step_return)
        print("CURRENT RISK : ", current_risk)
        print("ROLLING SHARPE RATIO : ", rolling_sharpe)
        print("PENALTY : ", penalty)
        print("MARGIN CALL PENALTY : ", margin_call_penalty)
        # print("MEAN RETURNS : ", np.mean(returns))
        print("LOG RETURNS BETWEEN LAST TWO TRADES : ", log_returns[-1])
        print("FINAL COMPOUNDED RETURN : ", final_compounded_return)
        print("SHARPE RATIO : ", sharpe_ratio)
        print("PROFIT / LOSS RATIO : ", profit_loss_ratio)
        print("PROFIT RISK RATIO : ", profit_risk_ratio)
        # print("TOTAL RETURN SINCE INITIAL BALANCE : ", final_return[-1])
        # print("SHARPE RATIO : ", sharpe_ratio)
        print("MAX DD : ", max_dd)
        print(
            "2 PERCENT OF BALANCE AND USED MARGIN : ",
            self.max_risk * (self.balance + self.position_value),
        )
        print("REWARD : ", reward)
        # self.episode_metrics["sharpe_ratios"].append(
        #     sharpe_ratio
        # )  # Log the Sharpe ratio
        return reward

    def log_step_metrics(self):

        if self.limit_bounds:
            # Log risk taken with stop loss levels
            if self.positions:
                position_type, entry_price, position_size, previous_leverage = (
                    self.positions[-1]
                )
                stop_loss_price, _ = self.stop_loss_levels[entry_price]
                current_risk = (
                    position_size * (abs(entry_price - stop_loss_price) / entry_price)
                ) / self.portfolio_value
                if not np.isnan(current_risk):
                    self.episode_metrics["risk_taken"].append(current_risk)

        # Log risk taken at each step
        if self.positions:
            position_type, entry_price, position_size, previous_leverage = (
                self.positions[-1]
            )
            current_risk = (
                position_size
                * (
                    (abs(entry_price - self.margin_price) / entry_price)
                    + self.liquidation_fee
                )
                + 2 * self.opening_fee
            ) / self.portfolio_value
            current_risk = min(current_risk, 1)
        else:
            current_risk = 0

        if not np.isnan(current_risk):
            self.episode_metrics["risk_taken"].append(current_risk)

        # Log drawdowns
        drawdown = calculate_max_drawdown(self.history)
        if not np.isnan(drawdown):
            self.episode_metrics["drawdowns"].append(drawdown)

        # Debugging: Log balance and positions
        print(
            f"Step: {self.current_step}, Balance: {self.balance}, Positions: {self.positions}, Portfolio Value: {self.portfolio_value}"
        )

    def log_episode_metrics(self):
        # Average metrics for the episode
        avg_return = (
            np.mean(self.episode_metrics["returns"])
            if self.episode_metrics["returns"]
            else 0
        )
        avg_risk_taken = (
            np.mean(self.episode_metrics["risk_taken"])
            if self.episode_metrics["risk_taken"]
            else 0
        )
        avg_sharpe_ratio = (
            np.mean(self.episode_metrics["sharpe_ratios"])
            if self.episode_metrics["sharpe_ratios"]
            else 0
        )
        avg_drawdown = (
            np.mean(self.episode_metrics["drawdowns"])
            if self.episode_metrics["drawdowns"]
            else 0
        )
        avg_leverage_used = (
            np.mean(self.episode_metrics["leverage_used"])
            if self.episode_metrics["leverage_used"]
            else 0
        )

        # Append to overall metrics
        self.metrics["returns"].append(avg_return)
        self.metrics["num_margin_calls"].append(
            self.episode_metrics["num_margin_calls"]
        )
        self.metrics["risk_taken"].append(avg_risk_taken)
        self.metrics["sharpe_ratios"].append(avg_sharpe_ratio)
        self.metrics["drawdowns"].append(avg_drawdown)
        self.metrics["num_trades"].append(self.episode_metrics["num_trades"])
        self.metrics["leverage_used"].append(avg_leverage_used)
        self.metrics["final_balance"].append(self.balance)

    def plot_episode_metrics(self):
        print("NUM TRADES: ", self.episode_metrics["num_trades"])
        # Plot the evolution of various metrics
        plt.figure(figsize=(14, 10))

        plt.subplot(5, 3, 1)
        plt.plot(self.episode_metrics["returns"], label="Returns")
        plt.title("Returns Over Steps")
        plt.legend()

        plt.subplot(5, 3, 2)
        plt.plot(self.episode_metrics["compounded_returns"], label="Compounded Returns")
        plt.title("Compounded Returns Over Steps")
        plt.legend()

        plt.subplot(5, 3, 3)
        plt.plot(self.episode_metrics["drawdowns"], label="Max Drawdown")
        plt.title("Max Drawdown Over Steps")
        plt.legend()

        plt.subplot(5, 3, 4)
        plt.plot(self.episode_metrics["risk_taken"], label="Risk Taken")
        plt.title("Risk Taken Over Steps")
        plt.legend()

        plt.subplot(5, 3, 5)
        plt.plot(self.episode_metrics["list_margin_calls"], label="Margin Calls")
        plt.title("Number of Margin Calls Over Steps")
        plt.legend()

        plt.subplot(5, 3, 6)
        plt.plot(self.episode_metrics["list_trades"], label="Number of Trades")
        plt.title("Number of Trades Over Steps")
        plt.legend()

        plt.subplot(5, 3, 7)
        plt.plot(self.episode_metrics["leverage_used"], label="Leverage Used")
        plt.title("Leverage Used Over Steps")
        plt.legend()

        plt.subplot(5, 3, 8)
        plt.plot(self.history, label="Balance")
        plt.title("Balance Over Steps")
        plt.legend()

        plt.subplot(5, 3, 9)
        plt.plot(self.episode_metrics["rewards"], label="Rewards")
        plt.title("Rewards Over Steps")
        plt.legend()

        plt.subplot(5, 3, 10)
        plt.plot(self.episode_metrics["sharpe_ratio"], label="Sharpe Ratio")
        plt.title("Sharpe ratio Over Steps")
        plt.legend()

        plt.subplot(5, 3, 11)
        plt.plot(self.episode_metrics["profit_risk_ratio"], label="P/R Ratio")
        plt.title("P/R ratio Over Steps")
        plt.legend()

        plt.subplot(5, 3, 12)
        plt.plot(self.episode_metrics["log_returns"], label="Log Returns")
        plt.title("Log Returns Over Steps")
        plt.legend()

        plt.subplot(5, 3, 13)
        plt.plot(
            self.episode_metrics["risk_adjusted_step_return"],
            label="Risk Adjusted Steps Log Returns",
        )
        plt.title("Log Returns Over Steps")
        plt.legend()

        plt.subplot(5, 3, 14)
        plt.plot(self.episode_metrics["rolling_sharpe"], label="Rolling Sharpe")
        plt.title("Log Returns Over Steps")
        plt.legend()

        plt.subplot(5, 3, 15)
        plt.plot(self.episode_metrics["step_risk"], label="Step Risk")
        plt.title("Log Returns Over Steps")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_metrics(self):
        # Plot the evolution of various metrics
        plt.figure(figsize=(14, 10))

        plt.subplot(3, 2, 1)
        plt.plot(self.metrics["returns"], label="Returns")
        plt.title("Returns Over Iterations")
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(self.metrics["sharpe_ratios"], label="Sharpe Ratio")
        plt.title("Sharpe Ratio Over Iterations")
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(self.metrics["drawdowns"], label="Max Drawdown")
        plt.title("Max Drawdown Over Iterations")
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(self.metrics["risk_taken"], label="Risk Taken")
        plt.title("Risk Taken Over Iterations")
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(self.metrics["num_margin_calls"], label="Margin Calls")
        plt.title("Number of Margin Calls Over Iterations")
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(self.metrics["num_trades"], label="Number of Trades")
        plt.title("Number of Trades Over Iterations")
        plt.legend()

        plt.subplot(3, 2, 7)
        plt.plot(self.metrics["leverage_used"], label="Leverage Used")
        plt.title("Leverage Used Over Iterations")
        plt.legend()

        plt.subplot(3, 2, 8)
        plt.plot(self.metrics["final_balance"], label="Final Balance")
        plt.title("Final Balance Over Iterations")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # def plot_prices(self):

    #     # self.data = hardcoded_dataset

    #     # Extract OHLC prices and indicators for the specified episode length
    #     open_prices = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 0
    #     ]
    #     high_prices = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 1
    #     ]
    #     low_prices = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 2
    #     ]
    #     close_prices = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 3
    #     ]

    #     # Extract precomputed indicators from the dataset
    #     ema20 = self.data[self.start_idx : self.start_idx + self.episode_length, -1, 5]
    #     ema50 = self.data[self.start_idx : self.start_idx + self.episode_length, -1, 6]
    #     ema100 = self.data[self.start_idx : self.start_idx + self.episode_length, -1, 7]
    #     bb_up_20 = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 8
    #     ]
    #     bb_low_20 = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 9
    #     ]
    #     bb_up_50 = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 10
    #     ]
    #     bb_low_50 = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 11
    #     ]
    #     atr14 = self.data[self.start_idx : self.start_idx + self.episode_length, -1, 12]
    #     atr50 = self.data[self.start_idx : self.start_idx + self.episode_length, -1, 13]
    #     rsi14 = self.data[self.start_idx : self.start_idx + self.episode_length, -1, 14]
    #     rsi30 = self.data[self.start_idx : self.start_idx + self.episode_length, -1, 15]
    #     macd = self.data[self.start_idx : self.start_idx + self.episode_length, -1, 16]
    #     signal = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 17
    #     ]
    #     plus_di_14 = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 18
    #     ]
    #     minus_di_14 = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 19
    #     ]
    #     adx14 = self.data[self.start_idx : self.start_idx + self.episode_length, -1, 20]
    #     plus_di_30 = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 21
    #     ]
    #     minus_di_30 = self.data[
    #         self.start_idx : self.start_idx + self.episode_length, -1, 22
    #     ]
    #     adx30 = self.data[self.start_idx : self.start_idx + self.episode_length, -1, 23]

    #     # Create a datetime index (adjust starting date as needed)
    #     date_range = pd.date_range(
    #         start="2019-11-01", periods=len(open_prices), freq="1H"
    #     )

    #     # Create a DataFrame for OHLC prices and indicators
    #     df = pd.DataFrame(
    #         {
    #             "Open": open_prices,
    #             "High": high_prices,
    #             "Low": low_prices,
    #             "Close": close_prices,
    #             "EMA20": ema20,
    #             "EMA50": ema50,
    #             "EMA100": ema100,
    #             "BB_up_20": bb_up_20,
    #             "BB_low_20": bb_low_20,
    #             "BB_up_50": bb_up_50,
    #             "BB_low_50": bb_low_50,
    #             "ATR14": atr14,
    #             "ATR50": atr50,
    #             "RSI14": rsi14,
    #             "RSI30": rsi30,
    #             "MACD": macd,
    #             "Signal": signal,
    #             "plus_di_14": plus_di_14,
    #             "minus_di_14": minus_di_14,
    #             "ADX14": adx14,
    #             "plus_di_30": plus_di_30,
    #             "minus_di_30": minus_di_30,
    #             "ADX30": adx30,
    #         },
    #         index=date_range,
    #     )

    #     # Create addplots for all the indicators
    #     addplots = [
    #         mpf.make_addplot(df["EMA20"], color="blue", width=1),
    #         mpf.make_addplot(df["EMA50"], color="green", width=1),
    #         mpf.make_addplot(df["EMA100"], color="red", width=1),
    #         mpf.make_addplot(df["BB_up_20"], color="orange", linestyle="--"),
    #         mpf.make_addplot(df["BB_low_20"], color="orange", linestyle="--"),
    #         mpf.make_addplot(df["BB_up_50"], color="purple", linestyle="--"),
    #         mpf.make_addplot(df["BB_low_50"], color="purple", linestyle="--"),
    #         mpf.make_addplot(df["ATR14"], panel=1, color="black", secondary_y=False),
    #         mpf.make_addplot(df["ATR50"], panel=1, color="grey", secondary_y=False),
    #         mpf.make_addplot(df["RSI14"], panel=2, color="blue", secondary_y=False),
    #         mpf.make_addplot(df["RSI30"], panel=2, color="green", secondary_y=False),
    #         mpf.make_addplot(df["MACD"], panel=3, color="purple", secondary_y=False),
    #         mpf.make_addplot(df["Signal"], panel=3, color="orange", secondary_y=False),
    #         mpf.make_addplot(
    #             df["plus_di_14"], panel=4, color="green", secondary_y=False
    #         ),
    #         mpf.make_addplot(
    #             df["minus_di_14"], panel=4, color="red", secondary_y=False
    #         ),
    #         mpf.make_addplot(df["ADX14"], panel=4, color="blue", secondary_y=False),
    #         mpf.make_addplot(
    #             df["plus_di_30"], panel=5, color="green", secondary_y=False
    #         ),
    #         mpf.make_addplot(
    #             df["minus_di_30"], panel=5, color="red", secondary_y=False
    #         ),
    #         mpf.make_addplot(df["ADX30"], panel=5, color="blue", secondary_y=False),
    #     ]

    #     # Plot candlestick chart with all indicators
    #     mpf.plot(
    #         df,
    #         type="candle",
    #         style="charles",
    #         title="Candlestick Chart with All Indicators",
    #         ylabel="Price",
    #         addplot=addplots,
    #         figscale=1.2,
    #         figratio=(16, 9),
    #         volume=False,
    #         panel_ratios=(3, 1, 1, 1, 1, 1),  # Adjust the height of each panel
    #     )

    def render(self):
        self.plot_episode_metrics()
        # self.plot_prices()


# Example functions for risk measures
def calculate_empirical_var(returns, confidence_level):
    if len(returns) == 0:
        return 0
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index]
    return var


def calculate_empirical_es(returns, confidence_level):
    if len(returns) == 0:
        return 0
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    tail_returns = sorted_returns[:index]
    es = -np.mean(tail_returns) if len(tail_returns) > 0 else 0
    return es


def calculate_max_drawdown(portfolio_values):
    if len(portfolio_values) == 0:
        return 0
    drawdowns = []
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak != 0 else 0
        drawdowns.append(drawdown)
    max_drawdown = max(drawdowns) if drawdowns else 0
    return max_drawdown


def calculate_skewness(returns):
    if len(returns) < 2:
        return 0
    std_returns = np.std(returns)
    if std_returns == 0:
        return 0
    return np.mean((returns - np.mean(returns)) ** 3) / std_returns**3


def calculate_kurtosis(returns):
    if len(returns) < 2:
        return 0
    std_returns = np.std(returns)
    if std_returns == 0:
        return 0
    return np.mean((returns - np.mean(returns)) ** 4) / std_returns**4
