import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
from ray.data import Dataset, from_numpy, read_parquet
import itertools
import json
import os
import gc
import cvxpy as cp

# Global variables

# min_size_usdt_assets = [
#     100,  # BTCUSDT placeholder
#     20,  # ETHUSDT placeholder
#     5,  # BNBUSDT placeholder
#     5,  # XRPUSDT placeholder
#     5,  # SOLUSDT placeholder
#     5,  # ADAUSDT placeholder
#     5,  # DOGEUSDT placeholder
#     5,  # TRXUSDT placeholder
#     5,  # AVAXUSDT placeholder
#     5,  # SHIBUSDT placeholder
#     5,  # DOTUSDT placeholder
# ]

min_size_usdt_assets = [
    5,  # ADAUSDT placeholder
    5,  # BNBUSDT placeholder
    100,  # BTCUSDT placeholder
    5,  # EOSUSDT placeholder
    20,  # ETHUSDT placeholder
    20,  # LTCUSDT placeholder
    5,  # NEOUSDT placeholder
    5,  # TRXUSDT placeholder
    5,  # XLMUSDT placeholder
    5,  # XRPUSDT placeholder
]

# min_trade_amount_assets = [
#     0.001,  # BTCUSDT min trade in BTC equivalent placeholder
#     0.001,  # ETHUSDT placeholder (or use notional checks)
#     0.01,  # BNBUSDT
#     0.1,  # XRPUSDT etc.
#     1,  # SOLUSDT
#     1,  # ADAUSDT
#     1,  # DOGEUSDT
#     1,  # TRXUSDT
#     1,  # AVAXUSDT
#     1,  # SHIBUSDT
#     0.1,  # DOTUSDT
# ]

min_trade_amount_assets = [
    1,  # ADAUSDT placeholder
    0.01,  # BNBUSDT placeholder
    0.001,  # BTCUSDT placeholder
    0.1,  # EOSUSDT placeholder
    0.001,  # ETHUSDT placeholder
    0.001,  # LTCUSDT placeholder
    0.01,  # NEOUSDT placeholder
    1,  # TRXUSDT placeholder
    1,  # XLMUSDT placeholder
    0.1,  # XRPUSDT placeholder
]

# max_market_amount_assets = [
#     120,  # BTCUSDT max amount in BTC or a large notional as placeholder
#     2000,  # ETHUSDT placeholder
#     2000,  # BNBUSDT placeholder
#     2000000,  # XRPUSDT placeholder
#     5000,  # SOLUSDT placeholder
#     300000,  # ADAUSDT placeholder
#     30000000,  # DOGEUSDT placeholder
#     5000000,  # TRXUSDT placeholder
#     5000,  # AVAXUSDT placeholder
#     50000000,  # SHIBUSDT placeholder
#     50000,  # DOTUSDT placeholder
# ]

max_market_amount_assets = [
    300000,  # ADAUSDT placeholder
    2000,  # BNBUSDT placeholder
    120,  # BTCUSDT placeholder
    120000,  # EOSUSDT placeholder
    2000,  # ETHUSDT placeholder
    5000,  # LTCUSDT placeholder
    10000,  # NEOUSDT placeholder
    5000000,  # TRXUSDT placeholder
    1000000,  # XLMUSDT placeholder
    2000000,  # XRPUSDT placeholder
]

# min_price_change_usdt = [
#     0.1,
#     0.01,
#     0.01,
#     0.0001,
#     0.01,
#     0.0001,
#     0.00001,
#     0.00001,
#     0.001,
#     0.000001,
#     0.001,
# ]

min_price_change_usdt = [
    0.0001,  # ADAUSDT placeholder
    0.01,  # BNBUSDT placeholder
    0.1,  # BTCUSDT placeholder
    0.001,  # EOSUSDT placeholder
    0.01,  # ETHUSDT placeholder
    0.01,  # LTCUSDT placeholder
    0.001,  # NEOUSDT placeholder
    0.00001,  # TRXUSDT placeholder
    0.00001,  # XLMUSDT placeholder
    0.0001,  # XRPUSDT placeholder
]


class TradingEnvironment(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path="train_portfolio_dataset_200_1d",
        input_length=200,
        market_fee=0.0005,
        limit_fee=0.0002,
        liquidation_fee=0.0125,
        slippage_mean=0.000001,
        slippage_std=0.00005,
        initial_balance=1000,
        total_episodes=1,
        episode_length=None,  # 24 hours of 5 minutes data
        max_risk=0.02,
        min_risk=0.001,
        min_profit=0,
        limit_bounds=False,
        render_mode=None,
    ):
        super(TradingEnvironment, self).__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Load your dataset
        self.data = read_parquet(data_path)

        # Number of assets in the portfolio
        # You must ensure that your dataset columns are arranged as explained:
        # For example, if you have 10 assets and each asset has 5 columns (Open,High,Low,Close,Volume),
        # your state should have these 50 columns for OHLCV (plus the additional static columns).
        self.num_assets = 10  # Adjust this number according to your actual dataset

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
        self.episode_length = self.data.count()
        # self.episode_length = episode_length
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
        self.limit_bounds = limit_bounds
        self.margin_price = 0
        self.current_position_size = 0
        self.current_risk = 0
        self.risk_adjusted_step = 0
        self.sharpe_ratio = 0
        self.new_margin_price = 0
        self.previous_leverage = 1

        # Initialize a positions dictionary for multiple assets
        # Each asset will have its own position data structure
        self.positions = {
            i: {
                "type": None,  # "long" or "short"
                "entry_price": 0.0,
                "size": 0.0,
                "leverage": 1,
            }
            for i in range(self.num_assets)
        }

        # Define the action space for multiple assets:
        # If limit_bounds=True: each asset has (weight, stop_loss, take_profit, leverage) = 4 parameters
        # If limit_bounds=False: each asset has (weight, leverage) = 2 parameters
        if self.limit_bounds:
            # shape = (num_assets * 4,)
            self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(self.num_assets * 4,),
                dtype=np.float32,
            )
        else:
            # shape = (num_assets * 2,)
            self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(self.num_assets,),
                dtype=np.float32,
            )

        # Initialize metrics dictionary if not present
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
            }

        self.sequence_buffer = []
        self.state, info = self.reset()

        # Initialize default static values depending on limit_bounds
        # These static values are appended at the end of each observation
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
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
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
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
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

        # #### Random method

        # # Randomly select a starting point in the dataset, ensuring there's enough data left for the episode
        # self.start_idx = random.randint(0, max_start_index - 1)

        # #### Iterative method

        # # Initialize or increment the starting index iteratively
        # if not hasattr(self, "current_start_idx"):
        #     self.current_start_idx = 0  # Start from the beginning initially
        # else:
        #     # Increment by 1 and wrap around if exceeding max_start_index
        #     self.current_start_idx = (self.current_start_idx + 1) % max_start_index

        # self.start_idx = self.current_start_idx

        #### One big episode method
        self.start_idx = 0

        self.current_step = self.start_idx

        self.unrealized_pnl = 0
        self.position_value = 0
        self.desired_position_size = 0
        self.compounded_returns = 1.0
        self.margin_call_triggered = False
        self.balance = self.initial_balance
        self.portfolio_value = self.balance
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
        self.action_history = []
        self.profits = []
        self.margin_price = 0
        self.current_position_size = 0
        self.current_risk = 0
        self.risk_adjusted_step = 0
        self.sharpe_ratio = 0
        self.new_margin_price = 0
        self.previous_leverage = 1

        # Reset positions for all assets
        self.positions = {
            i: {
                "type": None,  # "long" or "short"
                "entry_price": 0.0,
                "size": 0.0,
                "leverage": 1,
            }
            for i in range(self.num_assets)
        }

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
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
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
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
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
                prefetch_batches=10,
                drop_last=False,
            )
        )

        # self.iterator = itertools.islice(self.iterator, self.start_idx, None)

        self.sequence_buffer.clear()
        self.load_initial_sequences()

        self.state = self.sequence_buffer[0]

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
                next_state, info = self.reset()
                self.state = next_state
                terminated = True
                return self.state, terminated, info

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

        # Multi-asset action extraction
        # Assuming limit_bounds = False for simplicity, and that you receive [weight1, leverage1, weight2, leverage2, ..., weightN, leverageN]
        num_assets = self.num_assets
        # Parse the action depending on limit_bounds
        action = np.array(action)
        if self.limit_bounds:
            # action shape: (num_assets*4,)
            action = action.reshape(num_assets, 4)
            weights = action[:, 0]
            stop_losses = action[:, 1]
            take_profits = action[:, 2]
            leverages = action[:, 3]
        else:
            # action shape: (num_assets*2,)
            action = action.reshape(num_assets)
            weights = action
            default_leverage = 1
            # net_exposure = 1
            # stop_losses and take_profits are not used if limit_bounds=False
            stop_losses = np.zeros(num_assets)
            take_profits = np.zeros(num_assets)

        # Step 1: Enforce leverage constraint: sum(|w_i|) <= default_leverage
        total_leverage = np.sum(np.abs(weights))
        if (
            total_leverage > default_leverage and total_leverage > 0
        ):  # Avoid division by zero
            scaling_factor = default_leverage / total_leverage
            weights = weights * scaling_factor

        # # Step 2: Enforce net exposure constraint: |sum(w_i)| <= net_exposure
        # current_net_exposure = np.sum(weights)
        # if np.abs(current_net_exposure) > net_exposure:
        #     # Calculate the excess net exposure
        #     excess = np.abs(current_net_exposure) - net_exposure
        #     adjustment_direction = np.sign(current_net_exposure)

        #     if adjustment_direction > 0:
        #         # Too much long exposure, reduce positive weights
        #         positive_weights = weights > 0
        #         if np.sum(positive_weights) > 0:
        #             reduction_per_positive = excess / np.sum(positive_weights)
        #             weights[positive_weights] -= reduction_per_positive
        #     else:
        #         # Too much short exposure, reduce negative weights (increase toward zero)
        #         negative_weights = weights < 0
        #         if np.sum(negative_weights) > 0:
        #             reduction_per_negative = excess / np.sum(np.abs(negative_weights))
        #             weights[
        #                 negative_weights
        #             ] += reduction_per_negative  # Reduce magnitude

        #     # After adjustment, re-check leverage as it may have changed
        #     total_leverage = np.sum(np.abs(weights))
        #     if total_leverage > default_leverage and total_leverage > 0:
        #         scaling_factor = default_leverage / total_leverage
        #         weights = weights * scaling_factor

        # #### Optimizer method
        # # Decision variable: adjusted weights
        # n = self.num_assets
        # x = cp.Variable(n)
        # # Penalty parameter: a large value forces leverage toward 1
        # lambda_param = 1000.0
        # # The net exposure constraint: the sum of weights must equal 1.
        # constraints = [cp.sum(x) == 1]
        # # We want x to be as close as possible to the raw weights,
        # # but if the leverage (the L1 norm) exceeds 1, we add a penalty.
        # # Note: cp.pos(cp.norm1(x) - 1) is zero when cp.norm1(x) <= 1.
        # objective = cp.Minimize(
        #     cp.sum_squares(x - weights)
        #     + lambda_param * cp.square(cp.pos(cp.norm1(x) - 1))
        # )
        # # Solve the problem
        # prob = cp.Problem(objective, constraints)
        # result = prob.solve()
        # # Output the results
        # optimized_weights = x.value
        # weights = optimized_weights

        self.previous_action = weights
        self.action_history.append(
            {
                "Episode": self.current_episode,
                "Step": self.current_step,
                "Actions": weights.tolist(),
            }
        )

        # total_weight_sum = np.sum(np.abs(weights))  # if you mean absolute sum ≤ 1
        # # or np.sum(weights) if weights are guaranteed non-negative

        # max_allowed_sum = 1.0
        # if total_weight_sum > max_allowed_sum:
        #     # Scale all weights down proportionally
        #     weights = weights * (max_allowed_sum / total_weight_sum)

        # # Map leverages from [-1,1] to [1, self.max_leverage]
        # # mapped_leverage = ((value+1)/2)*(max_leverage-1) + 1
        # leverages = ((leverages + 1) / 2.0) * (self.max_leverage - 1) + 1
        # leverages = np.round(leverages).astype(int)
        # leverages = np.clip(leverages, 1, self.max_leverage)

        allowed_leverages = [None, None, None, None, None, None, None, None, None, None]
        desired_position_sizes = []
        current_position_sizes = []

        # Get current OHLCV for each asset
        # Assuming data layout: for asset i, columns: i*5 to i*5+4 (Open,High,Low,Close,Volume)
        current_opens = []
        current_highs = []
        current_lows = []
        current_closes = []
        current_volumes = []
        for i in range(num_assets):
            base_idx = i * 5
            current_open = self.state[-1, base_idx + 0]
            current_high = self.state[-1, base_idx + 1]
            current_low = self.state[-1, base_idx + 2]
            current_close = self.state[-1, base_idx + 3]
            current_volume = self.state[-1, base_idx + 4]

            current_opens.append(current_open)
            current_highs.append(current_high)
            current_lows.append(current_low)
            current_closes.append(current_close)
            current_volumes.append(current_volume)

        # Approximate bid/ask for each asset
        bids = []
        asks = []
        for i in range(num_assets):
            bid_i, ask_i = self.approximate_bid_ask(
                current_highs[i], current_lows[i], current_closes[i], current_volumes[i]
            )
            bids.append(bid_i)
            asks.append(ask_i)

        # Weighted by position size or notional
        total_size = sum(
            pos["size"] for pos in self.positions.values() if pos["size"] > 0
        )
        if total_size > 0:
            weighted_sum_bid = 0
            weighted_sum_ask = 0
            for i, pos in self.positions.items():
                if pos["size"] > 0:
                    # weight by pos["size"] or another factor
                    weighted_sum_bid += bids[i] * pos["size"]
                    weighted_sum_ask += asks[i] * pos["size"]

            aggregated_bid = weighted_sum_bid / total_size
            aggregated_ask = weighted_sum_ask / total_size
        else:
            aggregated_bid = np.mean(bids)  # Fallback if no positions
            aggregated_ask = np.mean(asks)

        self.current_bid = aggregated_bid  # reference
        self.current_ask = aggregated_ask

        # Compute a mark price simulation for each asset and take average
        mark_prices = []
        for i in range(num_assets):
            mark_std = self.get_std_dev_from_volume(current_volumes[i])
            normal_factor = np.random.normal(0, mark_std)
            mark_price_i = current_closes[i] * (1 + normal_factor)
            mark_prices.append(mark_price_i)
        self.mark_price = np.mean(mark_prices)

        # Reset realized_pnl and margin_fee for this step
        self.realized_pnl = 0
        self.margin_fee = 0

        # Compute initial unrealized PnL and position_value
        self.update_unrealized_pnl_all_assets(bids, asks)
        self.portfolio_value = self.balance + self.position_value + self.unrealized_pnl

        desired_position_sizes.clear()
        current_position_sizes.clear()

        # End of episode check: close all positions
        if (self.current_step + 1) >= (self.start_idx + self.episode_length) and any(
            pos["type"] for pos in self.positions.values()
        ):
            # Close all positions
            for i in range(num_assets):
                pos = self.positions[i]
                if pos["type"] is not None and pos["size"] > 0:
                    if pos["type"] == "long":
                        realized_pnl_i = pos["size"] * (
                            (bids[i] - pos["entry_price"]) / pos["entry_price"]
                        )
                        closing_fee = pos["size"] * self.market_fee
                        self.balance += (
                            realized_pnl_i - closing_fee - self.opening_fee
                        ) + (pos["size"] / pos["leverage"])
                        self.realized_pnl += (
                            realized_pnl_i - closing_fee - self.opening_fee
                        )
                    else:
                        realized_pnl_i = pos["size"] * (
                            (pos["entry_price"] - asks[i]) / pos["entry_price"]
                        )
                        closing_fee = pos["size"] * self.market_fee
                        self.balance += (
                            realized_pnl_i - closing_fee - self.opening_fee
                        ) + (pos["size"] / pos["leverage"])
                        self.realized_pnl += (
                            realized_pnl_i - closing_fee - self.opening_fee
                        )
                    # Reset position
                    self.positions[i] = {
                        "type": None,
                        "entry_price": 0.0,
                        "size": 0.0,
                        "leverage": 1,
                    }

            self.unrealized_pnl = 0
            reward = 0
            self.opening_fee = 0
            self.closing_fee = 0
            self.position_value = 0

        else:

            # Not end of episode, proceed with actions per asset
            reward = 0

            # Margin call check before acting
            self.check_margin_call(max(current_highs), min(current_lows))
            if self.limit_bounds:
                self.check_limits(current_highs, current_lows)

            # Randomize the order of asset execution to avoid bias
            asset_order = list(range(num_assets))
            np.random.shuffle(asset_order)

            # For each asset, apply the action
            epsilon = 1e-10
            for idx in asset_order:
                self.update_unrealized_pnl_all_assets(bids, asks)
                i = idx
                weight_i = weights[i]
                stop_loss_i = stop_losses[i]
                take_profit_i = take_profits[i]
                current_price = current_closes[i]
                pos = self.positions[i]

                # Determine final desired positions from weights
                total_equity = self.balance + self.position_value
                desired_position_i = weight_i * (total_equity * default_leverage)

                # Current position details
                current_size = pos["size"]
                current_position_sizes.append(current_size)
                current_direction = pos["type"]

                # difference_size determines how we adjust
                if current_direction == "long":
                    effective_current_size = current_size
                elif current_direction == "short":
                    effective_current_size = -current_size
                else:
                    effective_current_size = 0

                difference_size = desired_position_i - effective_current_size

                # Apply slippage to this asset's effective prices
                slippage = np.random.normal(self.slippage_mean, self.slippage_std)
                # If difference_size > 0: we are buying more (for long) or closing short (buy side) => use ask price with slippage
                # If difference_size < 0: we are selling (for long) or opening short => use bid price with slippage
                if difference_size > 0:
                    # Buying side => use ask price adjusted by slippage
                    effective_ask_price = asks[i] * (1 + slippage)
                    trade_price = (
                        round(effective_ask_price / min_price_change_usdt[i])
                        * min_price_change_usdt[i]
                    )
                else:
                    # Selling side => use bid price adjusted by slippage
                    effective_bid_price = bids[i] * (1 - slippage)
                    trade_price = (
                        round(effective_bid_price / min_price_change_usdt[i])
                        * min_price_change_usdt[i]
                    )

                # ------------------------------------------------------------
                # First handle position closing steps (no leverage/trade checks needed)
                # ------------------------------------------------------------

                # Handle partial/full closes first
                partially_closed = False
                fully_closed = False

                # Adjust positions:
                # If difference_size > 0 and we currently have a short, close it first
                if difference_size > 0 and current_direction == "short":
                    closing_size = min(difference_size, abs(effective_current_size))
                    if (closing_size) < abs(effective_current_size):

                        realized_pnl = closing_size * (
                            (pos["entry_price"] - trade_price) / pos["entry_price"]
                        )

                        self.opening_fee -= closing_size * self.market_fee
                        self.closing_fee = closing_size * self.market_fee
                        remaining_size = abs(effective_current_size) - closing_size

                        # Update balances
                        self.balance += (realized_pnl - 2 * self.closing_fee) + (
                            closing_size / pos["leverage"]
                        )
                        self.realized_pnl = realized_pnl - 2 * self.closing_fee
                        pos["size"] = remaining_size
                        pos["type"] = "short" if remaining_size > 0 else None
                        # After partial close: difference_size=0, no new position
                        difference_size = 0
                        partially_closed = True

                        # Update pnl and stop loss if needed
                        self.update_unrealized_pnl_all_assets(bids, asks)
                        if self.limit_bounds and pos["type"] is not None:
                            self.update_stop_loss_if_needed(
                                i, stop_loss_i, take_profit_i, mark_prices[i]
                            )

                        self.closing_fee = 0
                        self.consecutive_technical_miss = 0
                        self.consecutive_no_trade = 0

                    else:

                        # Close short at trade_price
                        pnl = current_size * (
                            (pos["entry_price"] - trade_price) / pos["entry_price"]
                        )
                        closing_fee = current_size * self.market_fee
                        self.balance += (pnl - closing_fee - self.opening_fee) + (
                            current_size / pos["leverage"]
                        )
                        self.realized_pnl += pnl - closing_fee - self.opening_fee
                        self.opening_fee = 0
                        pos["size"] = 0
                        pos["type"] = None
                        fully_closed = True
                        # Adjust difference_size by subtracting the closed size
                        # We used `abs(effective_current_size)` from difference_size
                        difference_size = difference_size - abs(effective_current_size)

                # If difference_size < 0 and we currently have a long, close it first
                if (
                    difference_size < 0
                    and current_direction == "long"
                    and not partially_closed
                    and not fully_closed
                ):
                    closing_size = min(
                        abs(difference_size), abs(effective_current_size)
                    )
                    if (closing_size) < abs(effective_current_size):

                        realized_pnl = closing_size * (
                            (trade_price - pos["entry_price"]) / pos["entry_price"]
                        )
                        self.opening_fee -= closing_size * self.market_fee
                        self.closing_fee = closing_size * self.market_fee
                        remaining_size = abs(effective_current_size) - closing_size

                        self.balance += (realized_pnl - 2 * self.closing_fee) + (
                            closing_size / pos["leverage"]
                        )
                        self.realized_pnl = realized_pnl - 2 * self.closing_fee
                        pos["size"] = remaining_size
                        pos["type"] = "long" if remaining_size > 0 else None
                        difference_size = 0  # after partial close no new position
                        partially_closed = True

                        self.update_unrealized_pnl_all_assets(bids, asks)
                        if self.limit_bounds and pos["type"] is not None:
                            self.update_stop_loss_if_needed(
                                i, stop_loss_i, take_profit_i, mark_prices[i]
                            )

                        self.closing_fee = 0
                        self.consecutive_technical_miss = 0
                        self.consecutive_no_trade = 0

                    else:
                        # Close long at trade_price
                        pnl = current_size * (
                            (trade_price - pos["entry_price"]) / pos["entry_price"]
                        )
                        closing_fee = current_size * self.market_fee
                        self.balance += (pnl - closing_fee - self.opening_fee) + (
                            current_size / pos["leverage"]
                        )
                        self.realized_pnl += pnl - closing_fee - self.opening_fee
                        self.opening_fee = 0
                        pos["size"] = 0
                        pos["type"] = None
                        fully_closed = True
                        # Adjust difference_size by subtracting the closed size
                        difference_size = difference_size + abs(
                            effective_current_size
                        )  # difference_size <0, add abs to increase difference_size

                if partially_closed or fully_closed:
                    # Partial close done, no new position open
                    # Move to next asset
                    self.consecutive_technical_miss = 0
                    self.consecutive_no_trade = 0
                    continue

                if difference_size == 0 and not fully_closed:
                    # No change in position size, possibly update stop-loss if needed
                    if pos["type"] is not None and self.limit_bounds:
                        self.update_stop_loss_if_needed(
                            i, stop_loss_i, take_profit_i, mark_prices[i]
                        )
                    else:
                        # No position & no trade => penalty no trade
                        penalty = min(
                            self.no_trade
                            + (0.5 * self.no_trade * self.consecutive_no_trade),
                            self.max_no_trade,
                        )
                        reward -= penalty
                        self.consecutive_no_trade += 1
                    continue

                if difference_size == 0 and fully_closed:
                    self.consecutive_technical_miss = 0
                    continue

                # Now handle opening new position (or increasing existing same-direction position)
                # For opening new position or increasing, we do margin/trade checks
                # difference_size!=0 at this point means no partial close was done

                # Check trade size constraints
                desired_unit_size = abs(difference_size) / current_price
                asset_min_size_usdt = min_size_usdt_assets[i]
                asset_min_trade_amount = min_trade_amount_assets[
                    i
                ]  # If needed, interpret as min units of asset
                asset_max_market_amount = max_market_amount_assets[i]

                # Check margin tier for allowed leverage
                allowed_leverage, mm_rate, mm_amount, _, _ = self.get_margin_tier(
                    i, abs(difference_size)
                )
                allowed_leverages[i] = allowed_leverage

                if default_leverage > allowed_leverage:
                    # Penalty for too high leverage
                    penalty = min(
                        self.technical_miss
                        + (0.5 * self.technical_miss * self.consecutive_technical_miss),
                        self.max_technical_miss,
                    )
                    reward -= penalty
                    self.consecutive_technical_miss += 1
                    # Skip this asset's position adjustment
                    continue

                if not (
                    abs(difference_size) >= asset_min_size_usdt
                    and desired_unit_size >= asset_min_trade_amount
                    and desired_unit_size <= asset_max_market_amount
                ):
                    # Trade not valid in size
                    penalty = min(
                        self.technical_miss
                        + (0.5 * self.technical_miss * self.consecutive_technical_miss),
                        self.max_technical_miss,
                    )
                    reward -= penalty
                    self.consecutive_technical_miss += 1
                    # Possibly update stop loss if limit_bounds and position exists
                    if self.limit_bounds and current_direction is not None:
                        self.update_stop_loss_if_needed(
                            i, stop_loss_i, take_profit_i, mark_prices[i]
                        )
                    continue

                required_margin = abs(difference_size) / default_leverage
                if self.balance < required_margin:
                    # Not enough balance
                    penalty = min(
                        self.technical_miss
                        + (0.5 * self.technical_miss * self.consecutive_technical_miss),
                        self.max_technical_miss,
                    )
                    reward -= penalty
                    self.consecutive_technical_miss += 1
                    continue

                # Now open/increase position in the direction of difference_size
                self.balance -= abs(difference_size) / default_leverage
                self.opening_fee += abs(difference_size) * self.market_fee

                new_size = abs(difference_size) + (
                    pos["size"]
                    if pos["type"] in ["long", "short"]
                    and (
                        (pos["type"] == "long" and difference_size > 0)
                        or (pos["type"] == "short" and difference_size < 0)
                    )
                    else 0
                )

                if difference_size > 0:
                    # Going long
                    if pos["type"] == "long":
                        # Weighted average entry
                        old_size = pos["size"]
                        new_entry_price = (
                            pos["entry_price"] * old_size
                            + trade_price * abs(difference_size)
                        ) / new_size
                        pos["entry_price"] = new_entry_price
                        pos["size"] = new_size
                        pos["leverage"] = default_leverage
                    else:
                        pos["type"] = "long"
                        pos["entry_price"] = trade_price
                        pos["size"] = abs(difference_size)
                        pos["leverage"] = default_leverage

                elif difference_size < 0:
                    # Going short
                    if pos["type"] == "short":
                        # Weighted average entry
                        old_size = pos["size"]
                        new_entry_price = (
                            pos["entry_price"] * old_size
                            + trade_price * abs(difference_size)
                        ) / new_size
                        pos["entry_price"] = new_entry_price
                        pos["size"] = new_size
                        pos["leverage"] = default_leverage
                    else:
                        pos["type"] = "short"
                        pos["entry_price"] = trade_price
                        pos["size"] = abs(difference_size)
                        pos["leverage"] = default_leverage

                # Update unrealized pnl after trade
                self.update_unrealized_pnl_all_assets(bids, asks)

                # If limit_bounds, set stop_loss/take_profit for this asset
                if self.limit_bounds and pos["type"] is not None:
                    self.update_stop_loss_if_needed(
                        i, stop_loss_i, take_profit_i, mark_prices[i]
                    )

                # Successful trade
                self.episode_metrics["num_trades"] += 1
                self.consecutive_technical_miss = 0
                self.consecutive_no_trade = 0

        # Recalculate portfolio metrics
        self.update_unrealized_pnl_all_assets(bids, asks)
        self.portfolio_value = self.balance + self.position_value + self.unrealized_pnl
        self.portfolio_value = round(self.portfolio_value, 5)
        self.history.append(self.portfolio_value)

        # Log metrics for the current step
        self.log_step_metrics()

        # Trading returns
        if len(self.history) > 1:
            if not self.margin_call_triggered:
                previous_val = self.history[-2]
                current_val = self.history[-1]
                trading_return = (
                    0
                    if previous_val == 0
                    else (current_val - previous_val) / previous_val
                )
                self.trading_returns.append(trading_return)
                self.episode_metrics["returns"].append(trading_return)
                log_trading_return = (
                    0
                    if (current_val <= 0 or previous_val <= 0)
                    else np.log(current_val / previous_val)
                )
                self.log_trading_returns.append(log_trading_return)
                final_return = (current_val - self.history[0]) / self.history[0]
                self.final_returns.append(final_return)
                reward += self.calculate_reward()
                self.episode_metrics["compounded_returns"].append(
                    self.compounded_returns - 1
                )
            else:
                # Margin call triggered this step
                reward += self.calculate_reward()
                self.episode_metrics["compounded_returns"].append(
                    self.compounded_returns - 1
                )
                self.margin_call_triggered = False
        else:
            reward += 0

        total_size = sum(
            pos["size"] for pos in self.positions.values() if pos["size"] > 0
        )
        if total_size > 0:

            allowed_leverage_count = 0
            sum_allowed_leverage = 0
            for pos in self.positions.values():
                if allowed_leverages[allowed_leverage_count] is not None:
                    sum_allowed_leverage += (
                        allowed_leverages[allowed_leverage_count] * pos["size"]
                    )
                    allowed_leverage_count += 1
            weighted_allowed_leverage = sum_allowed_leverage / total_size

            weighted_leverage = default_leverage

            weighted_entry_price = (
                sum(
                    pos["entry_price"] * pos["size"]
                    for pos in self.positions.values()
                    if pos["size"] > 0
                )
                / total_size
            )

            self.leverage = weighted_leverage
            self.allowed_leverage = weighted_allowed_leverage
            self.entry_price = weighted_entry_price
        else:
            self.leverage = 1
            self.allowed_leverage = 1
            self.entry_price = 0

        self.desired_position_size = np.sum(desired_position_sizes)
        self.current_position_size = np.sum(current_position_sizes)

        # Construct additional_state arrays
        last_log_ret = (
            self.log_trading_returns[-1] if len(self.log_trading_returns) > 0 else 0
        )
        if self.limit_bounds:
            additional_state = np.array(
                [
                    self.unrealized_pnl,
                    self.realized_pnl,
                    self.compounded_returns - 1,
                    self.take_profit_price if hasattr(self, "take_profit_price") else 0,
                    self.stop_loss_price if hasattr(self, "stop_loss_price") else 0,
                    self.entry_price,
                    self.leverage,
                    self.allowed_leverage,
                    self.balance,
                    self.position_value,
                    self.desired_position_size,
                    self.current_position_size,
                    last_log_ret,
                    self.previous_max_dd,
                    self.closing_fee,
                    self.opening_fee,
                    self.current_ask,
                    self.current_bid,
                    self.mark_price,
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
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
                    last_log_ret,
                    self.previous_max_dd,
                    self.closing_fee,
                    self.opening_fee,
                    self.current_ask,
                    self.current_bid,
                    self.mark_price,
                    self.current_risk,
                    self.risk_adjusted_step,
                    self.sharpe_ratio,
                ]
            )

        self.current_step += 1
        next_state, terminated, info = self.update_sequences(additional_state)
        self.state = next_state

        if self.portfolio_value < 0.1 * self.initial_balance:
            reward -= 1
            terminated = True

        if terminated:
            # Save the weights history at the end of the episode
            with open("weights_history.json", "w") as f:
                json.dump(self.action_history, f)
            self.state = next_state

        truncated = False

        return next_state, reward, terminated, truncated, info

    def set_limits(
        self, current_price, take_profit, position_type, adjusted_stop_loss, mark_price
    ):

        if not self.positions:
            return

        most_recent_position = self.positions[-1]
        entry_price = most_recent_position[1]

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

            self.stop_loss_levels[entry_price] = (stop_loss_price, position_type)
            self.take_profit_levels[entry_price] = (take_profit_price, position_type)
        else:
            # Only set stop-loss without take-profit
            self.stop_loss_levels[entry_price] = (stop_loss_price, position_type)
            # Optionally, you can still log that take-profit wasn't set

    def check_limits(self, high_price, low_price):

        if not self.positions:
            return

        most_recent_position = self.positions[-1]  # Get the most recent position
        entry_price = most_recent_position[1]

        if entry_price not in self.stop_loss_levels:
            return

        stop_loss_price, position_type = self.stop_loss_levels[entry_price]
        take_profit_price = None

        if entry_price in self.take_profit_levels:
            take_profit_price, _ = self.take_profit_levels[entry_price]

        if position_type == "long":
            if low_price <= stop_loss_price and (
                take_profit_price is not None and high_price >= take_profit_price
            ):
                execution_price = stop_loss_price
                self.episode_metrics["stop_loss_hits"] += 1  # Count stop-loss hit
                self.execute_order("sell", execution_price, entry_price)
                del self.stop_loss_levels[entry_price]
                del self.take_profit_levels[entry_price]

            elif low_price <= stop_loss_price or (
                take_profit_price is not None and high_price >= take_profit_price
            ):
                execution_price = (
                    stop_loss_price
                    if low_price <= stop_loss_price
                    else take_profit_price
                )
                if low_price <= stop_loss_price:
                    self.episode_metrics["stop_loss_hits"] += 1  # Count stop-loss hit
                self.execute_order("sell", execution_price, entry_price)
                del self.stop_loss_levels[entry_price]
                if take_profit_price is not None:
                    del self.take_profit_levels[entry_price]

            else:
                del self.stop_loss_levels[entry_price]
                if take_profit_price is not None:
                    del self.take_profit_levels[entry_price]

        elif position_type == "short":
            if high_price >= stop_loss_price and (
                take_profit_price is not None and low_price <= take_profit_price
            ):
                execution_price = stop_loss_price
                self.episode_metrics["stop_loss_hits"] += 1  # Count stop-loss hit
                self.execute_order("buy", execution_price, entry_price)
                del self.stop_loss_levels[entry_price]
                del self.take_profit_levels[entry_price]

            elif high_price >= stop_loss_price or (
                take_profit_price is not None and low_price <= take_profit_price
            ):
                execution_price = (
                    stop_loss_price
                    if high_price >= stop_loss_price
                    else take_profit_price
                )
                if high_price >= stop_loss_price:
                    self.episode_metrics["stop_loss_hits"] += 1  # Count stop-loss hit
                self.execute_order("buy", execution_price, entry_price)
                del self.stop_loss_levels[entry_price]
                if take_profit_price is not None:
                    del self.take_profit_levels[entry_price]

            else:
                del self.stop_loss_levels[entry_price]
                if take_profit_price is not None:
                    del self.take_profit_levels[entry_price]

    def execute_order(
        self,
        order_type,
        execution_price,
        entry_price,
        position_type,
        position_size,
        previous_leverage,
        is_margin_call=False,
    ):
        # Remove the logic that pops from self.positions since we now directly receive position details

        if not is_margin_call:
            # Normal position close (no margin call)
            if position_type == "long" and order_type == "sell":
                # Compute the realized profit
                profit_loss = position_size * (
                    (execution_price - entry_price) / entry_price
                )
                self.closing_fee = position_size * self.limit_fee
                self.balance += (profit_loss - self.closing_fee - self.opening_fee) + (
                    position_size / previous_leverage
                )
                self.realized_pnl = profit_loss - self.closing_fee - self.opening_fee

            elif position_type == "short" and order_type == "buy":
                # Compute the realized pnl
                profit_loss = position_size * (
                    (entry_price - execution_price) / entry_price
                )
                self.closing_fee = position_size * self.limit_fee
                self.balance += (profit_loss - self.closing_fee - self.opening_fee) + (
                    position_size / previous_leverage
                )
                self.realized_pnl = profit_loss - self.closing_fee - self.opening_fee

        else:
            # Position close due to margin call
            if position_type == "long" and order_type == "sell":
                realized_profit = position_size * (
                    (execution_price - entry_price) / entry_price
                )
                self.closing_fee = position_size * self.limit_fee
                margin_fee = position_size * self.liquidation_fee
                self.margin_fee = margin_fee
                self.realized_pnl = (
                    realized_profit - self.closing_fee - self.opening_fee - margin_fee
                )

                if (
                    self.balance
                    + self.realized_pnl
                    + (position_size / previous_leverage)
                ) < 0:
                    self.realized_pnl = -(
                        self.balance + (position_size / previous_leverage)
                    )

                self.balance = (
                    self.balance
                    + self.realized_pnl
                    + (position_size / previous_leverage)
                )

            elif position_type == "short" and order_type == "buy":
                realized_profit = position_size * (
                    (entry_price - execution_price) / entry_price
                )
                self.closing_fee = position_size * self.limit_fee
                margin_fee = position_size * self.liquidation_fee
                self.margin_fee = margin_fee
                self.realized_pnl = (
                    realized_profit - self.closing_fee - self.opening_fee - margin_fee
                )

                if (
                    self.balance
                    + self.realized_pnl
                    + (position_size / previous_leverage)
                ) < 0:
                    self.realized_pnl = -(
                        self.balance + (position_size / previous_leverage)
                    )

                self.balance = (
                    self.balance
                    + self.realized_pnl
                    + (position_size / previous_leverage)
                )

        # After closing a position:
        self.unrealized_pnl = 0
        self.position_value = 0
        self.portfolio_value = self.balance
        self.closing_fee = 0
        self.opening_fee = 0

    def calculate_liquidation_price(self):
        """
        Adapted calculation of a liquidation price for a multi-asset portfolio.
        This is a simplified approach:
        - Compute total notional value from all open positions.
        - Use that to get maintenance_margin_rate and required_margin.
        - Assume the 'worst-case scenario' is reflected by the first asset (or any chosen reference asset).
        For a real multi-asset scenario, a single liquidation price is not truly representative.
        """

        # If no positions, no margin price
        active_positions = [
            pos
            for pos in self.positions.values()
            if pos["type"] is not None and pos["size"] > 0
        ]
        if not active_positions:
            return 0

        # Compute total notional value
        total_notional = sum(pos["size"] for pos in active_positions)

        # Get margin tier from total_notional
        leverage, maintenance_margin_rate, maintenance_amount, _, _ = (
            self.get_margin_tier(total_notional)
        )
        required_margin = maintenance_margin_rate * total_notional - maintenance_amount

        # One-way mode, cross-margin logic: we need a reference asset for margin_price computation
        # Let's pick the first asset that has a position as reference:
        for i, pos in self.positions.items():
            if pos["type"] is not None and pos["size"] > 0:
                reference_asset_index = i
                reference_pos = pos
                break

        entry_price = reference_pos["entry_price"]
        position_type = reference_pos["type"]
        position_size = reference_pos["size"]

        # Calculate margin call price using reference asset logic
        # This mirrors the single asset logic but on the chosen reference position
        # margin_price = same formula as single asset:
        if position_type == "long":
            margin_price = (
                entry_price
                * (-(self.portfolio_value - required_margin) / position_size)
            ) + entry_price
        else:
            margin_price = entry_price - (
                entry_price
                * (-(self.portfolio_value - required_margin) / position_size)
            )

        margin_price = max(margin_price, 0)
        return margin_price

    def update_stop_loss_if_needed(
        self, asset_index, stop_loss, take_profit, mark_price
    ):
        """
        Update the stop loss and take profit for an asset's position if limit_bounds is True.
        This method assumes the position for this asset (self.positions[asset_index]) exists.
        It uses similar logic as in single asset scenario to ensure stop_loss is feasible.
        """

        if not self.limit_bounds:
            return  # No need to update stop loss if limit_bounds is False

        pos = self.positions[asset_index]
        if pos["type"] is None or pos["size"] <= 0:
            return  # No active position, no stop loss to update

        # Compute max_loss_per_unit and min_loss_per_unit, similar to single asset
        max_loss_per_unit = (
            self.max_risk * (self.balance + self.position_value)
            - (2 * self.opening_fee)
        ) / pos["size"]
        min_loss_per_unit = (
            self.min_risk * (self.balance + self.position_value)
            - (2 * self.opening_fee)
        ) / pos["size"]

        # Restricted loss calculation
        restricted_loss_per_unit = (
            1 - ((1 - self.cap_rate) * (mark_price / pos["entry_price"]))
            if pos["entry_price"] != 0
            else 0
        )

        # Compute final max
        final_max_loss_per_unit = min(max_loss_per_unit, restricted_loss_per_unit)

        # Adjust the stop_loss within the feasible range
        adjusted_stop_loss = max(
            min(stop_loss, final_max_loss_per_unit), min_loss_per_unit
        )

        if adjusted_stop_loss > 0 and adjusted_stop_loss >= min_loss_per_unit:
            # Set new limits based on position type
            self.set_limits(
                pos["entry_price"],
                take_profit,
                pos["type"],
                adjusted_stop_loss,
                mark_price,
            )

    def update_unrealized_pnl_all_assets(self, bids, asks):
        """
        Recalculate the unrealized PnL and position value for all assets after changes.
        This method updates self.unrealized_pnl and self.position_value based on current bids/asks.
        bids and asks are arrays of current bid/ask prices for each asset.
        """

        total_unrealized_pnl = 0
        total_position_value = 0
        for i, pos in self.positions.items():
            if pos["type"] is not None and pos["size"] > 0:
                if pos["type"] == "long":
                    upnl = (
                        pos["size"]
                        * ((bids[i] - pos["entry_price"]) / pos["entry_price"])
                        if pos["entry_price"] != 0
                        else 0
                    )
                else:  # short
                    upnl = (
                        pos["size"]
                        * ((pos["entry_price"] - asks[i]) / pos["entry_price"])
                        if pos["entry_price"] != 0
                        else 0
                    )

                pos_value = pos["size"] / pos["leverage"]
                total_unrealized_pnl += upnl
                total_position_value += pos_value

        self.unrealized_pnl = total_unrealized_pnl
        self.position_value = total_position_value

    def check_margin_call(self, high_price, low_price):
        if not self.positions:
            return

        total_required_margin = 0
        for asset_id, pos in self.positions.items():
            if pos["type"] is not None and pos["size"] > 0:
                notional_value = pos["size"]
                max_lev, mm_rate, mm_amt, low_s, up_s = self.get_margin_tier(
                    asset_id, notional_value
                )
                required_margin = mm_rate * notional_value - mm_amt
                if required_margin < 0:
                    required_margin = 0
                total_required_margin += required_margin

        current_portfolio_value = self.portfolio_value
        if current_portfolio_value < total_required_margin:
            # Margin call triggered
            any_long = any(p["type"] == "long" for p in self.positions.values())
            any_short = any(p["type"] == "short" for p in self.positions.values())

            if any_long and any_short:
                worst_case_price = (
                    low_price  # or pick whichever scenario you consider more realistic
                )
            elif any_long:
                worst_case_price = low_price
            else:
                worst_case_price = high_price

            self.handle_margin_call(worst_case_price)

    def handle_margin_call(self, worst_case_price):
        self.margin_call_triggered = True

        if not self.positions:
            return

        # Liquidate all positions
        for asset_id, position in list(self.positions.items()):
            position_type = position["type"]
            entry_price = position["entry_price"]
            position_size = position["size"]
            previous_leverage = position["leverage"]

            if position_type is not None and position_size > 0:
                order_type = "sell" if position_type == "long" else "buy"
                self.execute_order(
                    order_type=order_type,
                    execution_price=worst_case_price,
                    entry_price=entry_price,
                    position_type=position_type,
                    position_size=position_size,
                    previous_leverage=previous_leverage,
                    is_margin_call=True,
                )

        # After deleting all, re-init positions to maintain keys
        self.positions = {
            i: {"type": None, "entry_price": 0.0, "size": 0.0, "leverage": 1}
            for i in range(self.num_assets)
        }

        if self.limit_bounds:
            self.stop_loss_levels.clear()
            self.take_profit_levels.clear()

        self.episode_metrics["num_margin_calls"] += 1
        self.episode_metrics["list_margin_calls"].append(
            self.episode_metrics["num_margin_calls"]
        )

        # Log trading return (unchanged)
        if len(self.history) > 0:
            if self.history[-1] == 0:
                trading_return = 0
            else:
                trading_return = (
                    self.portfolio_value - self.history[-1]
                ) / self.history[-1]
            self.trading_returns.append(trading_return)
            self.episode_metrics["returns"].append(trading_return)
            if self.portfolio_value <= 0 or self.history[-1] <= 0:
                log_trading_return = 0
            else:
                log_trading_return = np.log(self.portfolio_value / self.history[-1])
            self.log_trading_returns.append(log_trading_return)
            final_return = (self.portfolio_value - self.history[0]) / self.history[0]
            self.final_returns.append(final_return)

    def get_margin_tier(self, asset_id, notional_value):
        # Store tiers in a dictionary keyed by asset_id
        # Each value is a list of tuples: (lower_bound, upper_bound, max_leverage, mm_rate, mm_amount)
        tiers = {
            0: [  # adausdt
                (0, 10000, 75, 0.005, 0),
                (10000, 50000, 50, 0.01, 50),
                (50000, 200000, 40, 0.015, 300),
                (200000, 1000000, 25, 0.02, 1300),
                (1000000, 2000000, 20, 0.025, 6300),
                (2000000, 10000000, 10, 0.05, 56300),
                (10000000, 20000000, 5, 0.1, 556300),
                (20000000, 25000000, 4, 0.125, 1056300),
                (25000000, 50000000, 2, 0.25, 4181300),
                (50000000, 100000000, 1, 0.5, 16681300),
            ],
            1: [  # bnbusdt
                (0, 10000, 75, 0.005, 0),
                (10000, 50000, 50, 0.01, 50),
                (50000, 200000, 40, 0.015, 300),
                (200000, 1000000, 25, 0.02, 1300),
                (1000000, 2000000, 20, 0.025, 6300),
                (2000000, 10000000, 10, 0.05, 56300),
                (10000000, 20000000, 5, 0.1, 556300),
                (20000000, 25000000, 4, 0.125, 1056300),
                (25000000, 50000000, 2, 0.25, 4181300),
                (50000000, 100000000, 1, 0.5, 16681300),
            ],
            2: [  # btcusdt
                (0, 50000, 125, 0.004, 0),
                (50000, 600000, 100, 0.005, 50),
                (600000, 3000000, 75, 0.0065, 950),
                (3000000, 12000000, 50, 0.01, 11450),
                (12000000, 70000000, 25, 0.02, 131450),
                (70000000, 100000000, 20, 0.025, 481450),
                (100000000, 230000000, 10, 0.05, 2981450),
                (230000000, 480000000, 5, 0.1, 14481450),
                (480000000, 600000000, 4, 0.125, 26481450),
                (600000000, 800000000, 3, 0.15, 41481450),
                (800000000, 1200000000, 2, 0.25, 121481450),
                (1200000000, 1800000000, 1, 0.5, 421481450),
            ],
            3: [  # eousdt
                (0, 10000, 75, 0.005, 0),
                (10000, 50000, 50, 0.01, 50),
                (50000, 200000, 40, 0.015, 300),
                (200000, 1000000, 25, 0.02, 1300),
                (1000000, 2000000, 20, 0.025, 6300),
                (2000000, 10000000, 10, 0.05, 56300),
                (10000000, 20000000, 5, 0.1, 556300),
                (20000000, 25000000, 4, 0.125, 1056300),
                (25000000, 50000000, 2, 0.25, 4181300),
                (50000000, 100000000, 1, 0.5, 16681300),
            ],
            4: [  # ethusdt
                (0, 50000, 125, 0.004, 0),
                (50000, 600000, 100, 0.005, 50),
                (600000, 3000000, 75, 0.0065, 950),
                (3000000, 12000000, 50, 0.01, 11450),
                (12000000, 50000000, 25, 0.02, 131450),
                (50000000, 65000000, 20, 0.025, 381450),
                (65000000, 150000000, 10, 0.05, 2006450),
                (150000000, 320000000, 5, 0.1, 9506450),
                (320000000, 400000000, 4, 0.125, 17506450),
                (400000000, 530000000, 3, 0.15, 27506450),
                (530000000, 800000000, 2, 0.25, 80506450),
                (800000000, 1200000000, 1, 0.5, 280506450),
            ],
            5: [  # ltcusdt
                (0, 10000, 75, 0.005, 0),
                (10000, 50000, 50, 0.01, 50),
                (50000, 200000, 40, 0.015, 300),
                (200000, 1000000, 25, 0.02, 1300),
                (1000000, 2000000, 20, 0.025, 6300),
                (2000000, 10000000, 10, 0.05, 56300),
                (10000000, 20000000, 5, 0.1, 556300),
                (20000000, 25000000, 4, 0.125, 1056300),
                (25000000, 50000000, 2, 0.25, 4181300),
                (50000000, 100000000, 1, 0.5, 16681300),
            ],
            6: [  # neousdt
                (0, 5000, 50, 0.006, 0),
                (5000, 50000, 25, 0.01, 20),
                (50000, 400000, 20, 0.025, 770),
                (400000, 800000, 10, 0.05, 10770),
                (800000, 1000000, 5, 0.1, 50770),
                (1000000, 1500000, 4, 0.125, 75770),
                (1500000, 2000000, 2, 0.25, 263270),
                (2000000, 2500000, 1, 0.5, 763270),
            ],
            7: [  # trxusdt
                (0, 10000, 75, 0.0065, 0),
                (10000, 90000, 50, 0.01, 35),
                (90000, 120000, 40, 0.015, 485),
                (120000, 650000, 25, 0.02, 1085),
                (650000, 800000, 20, 0.025, 4335),
                (800000, 3000000, 10, 0.05, 24335),
                (3000000, 6000000, 5, 0.1, 174335),
                (6000000, 12000000, 4, 0.125, 324335),
                (12000000, 20000000, 2, 0.25, 1824335),
                (20000000, 30000000, 1, 0.5, 6824335),
            ],
            8: [  # xlmusdt
                (0, 10000, 75, 0.01, 0),
                (10000, 100000, 50, 0.015, 50),
                (100000, 500000, 25, 0.02, 550),
                (500000, 1000000, 20, 0.025, 3050),
                (1000000, 5000000, 10, 0.05, 28050),
                (5000000, 10000000, 5, 0.1, 278050),
                (10000000, 12500000, 4, 0.125, 528050),
                (12500000, 25000000, 2, 0.25, 2090550),
                (25000000, 50000000, 1, 0.5, 8340550),
            ],
            9: [  # xrpusdt
                (0, 10000, 75, 0.005, 0),
                (10000, 20000, 50, 0.0065, 15),
                (20000, 160000, 40, 0.01, 85),
                (160000, 1000000, 25, 0.02, 1685),
                (1000000, 2000000, 20, 0.025, 6685),
                (2000000, 10000000, 10, 0.05, 56685),
                (10000000, 20000000, 5, 0.1, 556685),
                (20000000, 25000000, 4, 0.125, 1056685),
                (25000000, 50000000, 2, 0.25, 4181685),
                (50000000, 100000000, 1, 0.5, 16681685),
            ],
        }

        # Retrieve the tier list for the given asset_id
        tier_list = tiers[asset_id]
        # Iterate through tiers to find the one matching the notional_value
        for low, high, max_lev, mm_rate, mm_amt in tier_list:
            if low <= notional_value <= high:
                return max_lev, mm_rate, mm_amt, low, high

        # If notional_value exceeds the highest tier, return the last tier's values
        last_tier = tier_list[-1]
        return last_tier[2], last_tier[3], last_tier[4], last_tier[0], last_tier[1]

    def calculate_reward(self):
        if len(self.history) < 2:
            return 0  # Not enough data to calculate reward

        returns = np.array(self.trading_returns)
        if len(returns) < 1 or np.isnan(returns).any():
            return 0  # Not enough data to calculate returns

        # max_dd = calculate_max_drawdown(self.history)

        # # Rolling sharpe ratio
        # WINDOW_SIZE = min(len(returns), 30)
        # RISK_FREE_RATE = 0.005 / (365.25 * 24)
        # ROLLING_SHARPE_FACTOR = WINDOW_SIZE / 30

        # window_returns = returns[-WINDOW_SIZE:]
        # mean_return = np.mean(window_returns)
        # std_return = np.std(window_returns)
        # rolling_sharpe = (
        #     ((mean_return - RISK_FREE_RATE) / std_return) if std_return != 0 else 0
        # )
        # self.sharpe_ratio = rolling_sharpe

        # Add log-returns as reward
        log_returns = np.array(self.log_trading_returns)
        if len(log_returns) < 1 or np.isnan(log_returns).any():
            return 0  # Not enough data for log returns

        # Compute current risk from all open positions
        # For each asset, compute a risk component similar to single asset logic and sum or take max.
        # We'll sum the risk from each asset's position.
        # current_risk = 0
        # if self.positions:
        #     for asset, pos in self.positions.items():
        #         if pos["type"] is not None and pos["size"] > 0:
        #             # Assume margin_price and liquidation_fee, opening_fee apply globally or per asset
        #             # If margin_price is portfolio-level, we approximate asset-level risk similarly
        #             # Using the same formula: ((abs(entry_price - margin_price)/entry_price)+liquidation_fee)*size +2*opening_fee)/portfolio_value
        #             # margin_price is a single number. For multi-assets, this is an approximation.
        #             # Alternatively, you could define a per-asset margin_price or similar logic.
        #             entry_price = pos["entry_price"]
        #             position_size = pos["size"]
        #             # Using the global margin_price as a proxy for all assets:
        #             asset_risk = (
        #                 (abs(entry_price - self.margin_price) / entry_price)
        #                 + self.liquidation_fee
        #             ) * position_size + (2 * self.opening_fee)
        #             current_risk += asset_risk
        #     # Normalize by portfolio value
        #     current_risk = (
        #         current_risk / self.portfolio_value if self.portfolio_value != 0 else 1
        #     )
        #     current_risk = min(current_risk, 1)
        # else:
        #     current_risk = 0

        # self.current_risk = current_risk

        # total_required_margin = 0
        # for asset_id, pos in self.positions.items():
        #     if pos["type"] is not None and pos["size"] > 0:
        #         notional_value = pos["size"]
        #         max_lev, mm_rate, mm_amt, low_s, up_s = self.get_margin_tier(
        #             asset_id, notional_value
        #         )
        #         required_margin = mm_rate * notional_value - mm_amt
        #         if required_margin < 0:
        #             required_margin = 0
        #         total_required_margin += required_margin

        # if self.portfolio_value > 0:
        #     self.current_risk = min(total_required_margin / self.portfolio_value, 1)
        # else:
        #     self.current_risk = 1

        # # Adjust step return by current risk
        # if self.current_risk > 0:
        #     risk_adjusted_step_return = log_returns[-1] / self.current_risk
        # else:
        #     risk_adjusted_step_return = log_returns[-1]

        # self.risk_adjusted_step = risk_adjusted_step_return

        # penalty = 0
        # if max_dd > self.previous_max_dd:
        #     penalty -= max_dd
        # self.previous_max_dd = max_dd

        # # Update compounded returns
        # self.compounded_returns *= 1 + returns[-1]
        # final_compounded_return = self.compounded_returns - 1

        # end_episode_comp_return = 0
        # if (self.current_step + 1) >= (self.start_idx + self.episode_length):
        #     end_episode_comp_return = final_compounded_return

        # # If you track leverage in a multi-asset scenario, self.previous_leverage may need updating for multiple assets.
        # # For simplicity, assume self.previous_leverage is updated elsewhere as an average or last used leverage.
        # leverage_factor = 0.001 * self.previous_leverage
        # leverage_bonus = 0
        # if log_returns[-1] > leverage_factor:
        #     leverage_bonus = self.previous_leverage * 0.01

        #### Original Sharpe ratio
        # if (self.current_step + 1) >= (self.start_idx + self.episode_length):
        #     mean_returns = np.mean(returns)
        #     std_returns = np.std(returns)
        #     risk_free_rate = 0.005 / (365 * 24)
        #     sharpe_ratio = (
        #         (mean_returns - risk_free_rate) / std_returns if std_returns != 0 else 0
        #     )
        # else:
        #     sharpe_ratio = 0

        #### Original Min Variance
        # Parameters
        rolling_window = 10  # Adjustable window size for rolling std
        alpha = 0.05  # Weight for log returns in step-wise reward
        lambda_ = 0.1  # Weight for rolling std penalty in step-wise reward
        beta = 1.0  # Weight for intermediate std penalty at truncation points
        gamma = 2.0  # Weight for final std penalty at the end of the full episode

        # Total_steps is the total number of steps in the full episode
        total_steps = self.episode_length
        # Reward calculation
        if (self.current_step + 1) >= (self.start_idx + self.episode_length):
            # End of the full episode
            std_returns = np.std(
                returns
            )  # Standard deviation of returns over the full episode
            reward = -std_returns * gamma  # Final penalty with higher weight
        if (self.current_step + 1) % 120 == 0:
            # End of a truncated episode (every 120 steps)
            std_returns_so_far = np.std(returns)  # Std of returns accumulated so far
            scaling_factor = (
                self.current_step + 1
            ) / total_steps  # Scales penalty by progress
            reward = -std_returns_so_far * beta * scaling_factor  # Intermediate penalty
        else:
            # # In between steps
            # rolling_std = (
            #     np.std(returns[-rolling_window:])
            #     if len(returns) >= rolling_window
            #     else 0
            # )
            # reward = (log_returns[-1] * alpha) - (
            #     rolling_std * lambda_
            # )  # Step-wise reward
            reward = log_returns[-1] * alpha
            # reward = 0

        # #### Original Sharpe ratio
        # # Parameters
        # alpha = 1.0  # Weight for log returns in step-wise reward
        # beta = 1.0  # Weight for intermediate sharpe ratio at truncation points
        # gamma = 2.0  # Weight for final sharpe ratio at the end of the full episode

        # # Total_steps is the total number of steps in the full episode
        # total_steps = self.episode_length
        # # Reward calculation
        # if (self.current_step + 1) >= (self.start_idx + self.episode_length):
        #     # End of the full episode
        #     mean_returns = np.mean(returns)
        #     std_returns = np.std(returns)
        #     risk_free_rate = 0.005 / (365 * 24)
        #     sharpe_ratio = (
        #         (mean_returns - risk_free_rate) / std_returns if std_returns != 0 else 0
        #     )
        #     reward = sharpe_ratio * gamma  # Final penalty with higher weight
        # if (self.current_step + 1) % 120 == 0:
        #     # End of a truncated episode (every 120 steps)
        #     mean_returns = np.mean(returns)
        #     std_returns = np.std(returns)
        #     risk_free_rate = 0.005 / (365 * 24)
        #     sharpe_ratio_so_far = (
        #         (mean_returns - risk_free_rate) / std_returns if std_returns != 0 else 0
        #     )
        #     scaling_factor = (
        #         self.current_step + 1
        #     ) / total_steps  # Scales penalty by progress
        #     reward = sharpe_ratio_so_far * beta * scaling_factor  # Intermediate penalty
        # else:
        #     # In between steps
        #     # rolling_std = (
        #     #     np.std(returns[-rolling_window:])
        #     #     if len(returns) >= rolling_window
        #     #     else 0
        #     # )
        #     # reward = (log_returns[-1] * alpha) - (
        #     #     rolling_std * lambda_
        #     # )  # Step-wise reward
        #     reward = log_returns[-1] * alpha

        # reward = (
        #     log_returns[-1]
        #     # - (std_returns * 1.0)
        #     # sharpe_ratio
        #     # log_returns[-1]
        #     # + self.risk_adjusted_step * 0.5
        #     # + rolling_sharpe * ROLLING_SHARPE_FACTOR
        #     # + end_episode_comp_return * 3
        #     # + leverage_bonus
        # )

        return reward

    def log_step_metrics(self):
        if self.limit_bounds and self.positions:
            # Compute risk taken with stop loss levels for each asset and average or sum them
            # We'll just append the average risk taken across all positions with a stop loss
            total_risk_taken = []
            for asset, pos in self.positions.items():
                if pos["type"] is not None and pos["size"] > 0:
                    entry_price = pos["entry_price"]
                    # Assume stop_loss_levels is keyed by (asset, entry_price)
                    if (asset, entry_price) in self.stop_loss_levels:
                        stop_loss_price, _ = self.stop_loss_levels[(asset, entry_price)]
                        current_risk = (
                            pos["size"]
                            * (abs(entry_price - stop_loss_price) / entry_price)
                        ) / self.portfolio_value
                        if not np.isnan(current_risk):
                            total_risk_taken.append(current_risk)
            if total_risk_taken:
                avg_risk = np.mean(total_risk_taken)
                self.episode_metrics["risk_taken"].append(avg_risk)

        # Log drawdowns
        drawdown = calculate_max_drawdown(self.history)
        if not np.isnan(drawdown):
            self.episode_metrics["drawdowns"].append(drawdown)

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
        plt.figure(figsize=(14, 10))

        plt.subplot(4, 2, 1)
        plt.plot(self.episode_metrics["returns"], label="Returns")
        plt.title("Returns Over Steps")
        plt.legend()

        plt.subplot(4, 2, 2)
        plt.plot(self.episode_metrics["compounded_returns"], label="Compounded Returns")
        plt.title("Compounded Returns Over Steps")
        plt.legend()

        plt.subplot(4, 2, 3)
        plt.plot(self.episode_metrics["drawdowns"], label="Max Drawdown")
        plt.title("Max Drawdown Over Steps")
        plt.legend()

        plt.subplot(4, 2, 4)
        plt.plot(self.episode_metrics["risk_taken"], label="Risk Taken")
        plt.title("Risk Taken Over Steps")
        plt.legend()

        plt.subplot(4, 2, 5)
        plt.plot(self.episode_metrics["list_margin_calls"], label="Margin Calls")
        plt.title("Number of Margin Calls Over Steps")
        plt.legend()

        plt.subplot(4, 2, 6)
        plt.plot(self.episode_metrics["list_trades"], label="Number of Trades")
        plt.title("Number of Trades Over Steps")
        plt.legend()

        plt.subplot(4, 2, 7)
        plt.plot(self.episode_metrics["leverage_used"], label="Leverage Used")
        plt.title("Leverage Used Over Steps")
        plt.legend()

        plt.subplot(4, 2, 8)
        plt.plot(self.history, label="Balance")
        plt.title("Balance Over Steps")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_metrics(self):
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

    def render(self):
        self.plot_episode_metrics()


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
