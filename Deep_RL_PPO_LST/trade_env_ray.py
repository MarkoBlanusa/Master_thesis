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


class TradingEnvironment(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        # data: Dataset,
        input_length=40,
        market_fee=0.0005,
        limit_fee=0.0002,
        liquidation_fee=0.0125,
        slippage_mean=0.000001,
        slippage_std=0.00005,
        initial_balance=1000,
        total_episodes=1,
        episode_length=168,  # 24 hours of 5 minutes data
        max_risk=0.02,
        min_risk=0.001,
        min_profit=0,
        limit_bounds=False,
        render_mode=None,
    ):
        super(TradingEnvironment, self).__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # project_dir = r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_1ère\2ème_semestre\ADA_2.0\Source_code_ADA"
        # hardcoded_dataset = np.load(os.path.join(project_dir, "train_data.npy"))

        self.data = read_parquet("train_dataset_40_1h")
        # del hardcoded_dataset
        # gc.collect()

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
            }

        self.sequence_buffer = []
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

        # Randomly select a starting point in the dataset, ensuring there's enough data left for the episode
        self.start_idx = random.randint(0, max_start_index - 1)
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
        self.action_history = []
        self.profits = []
        self.margin_price = 0
        self.current_position_size = 0
        self.current_risk = 0
        self.risk_adjusted_step = 0
        self.sharpe_ratio = 0
        self.new_margin_price = 0
        self.previous_leverage = 1

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

        if self.limit_bounds:
            weight, stop_loss, take_profit, leverage = action
        else:
            weight, leverage = action
        self.previous_action = action

        self.action_history.append(
            {
                "Episode": self.current_episode,
                "Step": self.current_step,
                "Actions": action.tolist(),
            }
        )

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
        current_volume = self.state[-1, 4]

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

        weight_diff = weight - current_weight

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
            self.desired_position_size = desired_position_size

            # Check the position size to see if it fits the margin rules and adapt the optimal leverage
            allowed_leverage, _, _, lower_size, upper_size = self.get_margin_tier(
                abs(desired_position_size)
            )
            self.allowed_leverage = allowed_leverage

            if leverage > allowed_leverage:
                penalty = min(
                    self.technical_miss
                    + (0.5 * self.technical_miss * self.consecutive_technical_miss),
                    self.max_technical_miss,
                )
                reward -= penalty
                self.consecutive_technical_miss += 1

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

                        adjusted_stop_loss = max(
                            min(stop_loss, final_max_loss_per_unit), min_loss_per_unit
                        )

                        # Set new limits even if the position is the same
                        self.set_limits(
                            self.positions[-1][1],
                            take_profit,
                            self.positions[-1][0],
                            adjusted_stop_loss,
                            mark_price,
                        )

            else:
                # Save the leverage if it is fitting the margin criterias
                self.leverage = leverage

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

                # Calculate the difference in size based on the position type
                difference_size = desired_position_size - current_position_size

                # Compute the desired size in BTC
                desired_btc_size = abs(desired_position_size) / current_price

                if (
                    abs(desired_position_size) >= self.min_size_usdt
                    and desired_btc_size >= self.min_trade_btc_amount
                    and desired_btc_size <= self.max_market_btc_amount
                ):

                    if difference_size > 0:  # Increase position

                        # Increase the long position
                        if current_position_size > 0:

                            difference_size_bound = abs(difference_size)

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

                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
                                        )

                                        if (
                                            adjusted_stop_loss > 0
                                            and adjusted_stop_loss >= min_loss_per_unit
                                        ):

                                            # Update the balance
                                            self.balance -= required_margin

                                            # Update the new increased position
                                            self.positions[-1] = (
                                                "long",
                                                new_entry_price,
                                                new_size,
                                                combined_leverage,
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

                                            adjusted_stop_loss = max(
                                                min(stop_loss, final_max_loss_per_unit),
                                                min_loss_per_unit,
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

                                    else:
                                        # Update the balance
                                        self.balance -= required_margin

                                        # Update the new increased position
                                        self.positions[-1] = (
                                            "long",
                                            new_entry_price,
                                            new_size,
                                            combined_leverage,
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

                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
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

                        # Diminishing the short position and potentitally closing it to open a long position
                        if current_position_size < 0:  # Closing a short position

                            difference_size = abs(difference_size)

                            # Ensure to not close more than the current short position
                            closing_size = min(
                                difference_size, abs(current_position_size)
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

                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
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

                                            # Calculate and deduct the transaction fee
                                            self.balance += (
                                                realized_pnl - 2 * self.closing_fee
                                            ) + (closing_size / self.positions[0][3])

                                            self.realized_pnl = (
                                                realized_pnl - 2 * self.closing_fee
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

                                            adjusted_stop_loss = max(
                                                min(stop_loss, final_max_loss_per_unit),
                                                min_loss_per_unit,
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

                                    else:
                                        self.positions[0] = (
                                            self.positions[0][0],
                                            self.positions[0][1],
                                            remaining_size,
                                            self.positions[0][3],
                                        )

                                        # Calculate and deduct the transaction fee
                                        self.balance += (
                                            realized_pnl - 2 * self.closing_fee
                                        ) + (closing_size / self.positions[0][3])

                                        self.realized_pnl = (
                                            realized_pnl - 2 * self.closing_fee
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

                        # Open a new long position with the exceeding size
                        if (
                            current_position_size <= 0
                            and (abs(difference_size) - abs(current_position_size)) > 0
                            and desired_position_size > 0
                        ):
                            new_position_size = abs(difference_size) - abs(
                                current_position_size
                            )
                            required_margin = new_position_size / self.leverage

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

                                    adjusted_stop_loss = max(
                                        min(stop_loss, final_max_loss_per_unit),
                                        min_loss_per_unit,
                                    )

                                    if (
                                        adjusted_stop_loss > 0
                                        and adjusted_stop_loss >= min_loss_per_unit
                                    ):

                                        self.balance -= required_margin

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

                                else:

                                    self.balance -= required_margin

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

                    elif difference_size < 0:  # Decrease position

                        # Increase the short position
                        if current_position_size < 0:

                            difference_size_bound = abs(difference_size)

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

                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
                                        )

                                        if (
                                            adjusted_stop_loss > 0
                                            and adjusted_stop_loss >= min_loss_per_unit
                                        ):

                                            self.balance -= required_margin

                                            # Update the new increased position
                                            self.positions[-1] = (
                                                "short",
                                                new_entry_price,
                                                new_size,
                                                combined_leverage,
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

                                            adjusted_stop_loss = max(
                                                min(stop_loss, final_max_loss_per_unit),
                                                min_loss_per_unit,
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

                                    else:

                                        self.balance -= required_margin

                                        # Update the new increased position
                                        self.positions[-1] = (
                                            "short",
                                            new_entry_price,
                                            new_size,
                                            combined_leverage,
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

                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
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

                        # Diminishing the long position and potentitally closing it to open a short position
                        if current_position_size > 0:  # Closing a long position

                            difference_size = abs(difference_size)

                            # Ensure to not close more than the current long position
                            closing_size = min(
                                difference_size, abs(current_position_size)
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

                                        adjusted_stop_loss = max(
                                            min(stop_loss, final_max_loss_per_unit),
                                            min_loss_per_unit,
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

                                            # Calculate and deduct the transaction fee
                                            self.balance += (
                                                realized_pnl - 2 * self.closing_fee
                                            ) + (closing_size / self.positions[-1][3])

                                            self.realized_pnl = (
                                                realized_pnl - 2 * self.closing_fee
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

                                            adjusted_stop_loss = max(
                                                min(stop_loss, final_max_loss_per_unit),
                                                min_loss_per_unit,
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

                                    else:

                                        self.positions[0] = (
                                            self.positions[0][0],
                                            self.positions[0][1],
                                            remaining_size,
                                            self.positions[-1][3],
                                        )

                                        # Calculate and deduct the transaction fee
                                        self.balance += (
                                            realized_pnl - 2 * self.closing_fee
                                        ) + (closing_size / self.positions[-1][3])

                                        self.realized_pnl = (
                                            realized_pnl - 2 * self.closing_fee
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

                        # Open a new short position with the remaining size
                        if (
                            current_position_size >= 0
                            and (abs(difference_size) - abs(current_position_size)) > 0
                            and desired_position_size < 0
                        ):
                            new_position_size = abs(difference_size) - abs(
                                current_position_size
                            )

                            required_margin = new_position_size / self.leverage

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

                                    adjusted_stop_loss = max(
                                        min(stop_loss, final_max_loss_per_unit),
                                        min_loss_per_unit,
                                    )

                                    if (
                                        adjusted_stop_loss > 0
                                        and adjusted_stop_loss >= min_loss_per_unit
                                    ):

                                        self.balance -= required_margin

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

                                else:

                                    self.balance -= required_margin

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

                                adjusted_stop_loss = max(
                                    min(stop_loss, final_max_loss_per_unit),
                                    min_loss_per_unit,
                                )

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

                            else:
                                # Penalize the model
                                penalty = min(
                                    self.no_trade
                                    + (0.5 * self.no_trade * self.consecutive_no_trade),
                                    self.max_no_trade,
                                )
                                reward -= penalty
                                self.consecutive_no_trade += 1

                        else:
                            # Penalize the model
                            penalty = min(
                                self.no_trade
                                + (0.5 * self.no_trade * self.consecutive_no_trade),
                                self.max_no_trade,
                            )
                            reward -= penalty
                            self.consecutive_no_trade += 1

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

                            adjusted_stop_loss = max(
                                min(stop_loss, final_max_loss_per_unit),
                                min_loss_per_unit,
                            )

                            # Set new limits even if the position is the same
                            self.set_limits(
                                self.positions[-1][1],
                                take_profit,
                                self.positions[-1][0],
                                adjusted_stop_loss,
                                mark_price,
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

        self.portfolio_value = self.balance + self.position_value
        self.portfolio_value = round(self.portfolio_value, 5)
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

        # Compute the new margin price
        new_margin_price = self.calculate_liquidation_price()
        self.new_margin_price = new_margin_price

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

        self.current_step += 1
        next_state, terminated, info = self.update_sequences(additional_state)
        self.state = next_state

        if self.portfolio_value < 0.1 * self.initial_balance:
            reward -= 100
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
        self, order_type, execution_price, entry_price, margin_price=None
    ):

        # Apply slippage to the execution price
        # slippage = np.random.normal(self.slippage_mean, self.slippage_std)
        # execution_price *= 1 + slippage if order_type == "buy" else 1 - slippage

        if not self.positions:
            return

        position = self.positions.pop(0)
        position_type, entry_price, position_size, previous_leverage = position

        if not self.margin_call_triggered:
            if position_type == "long" and order_type == "sell":

                # Compute the realized profit
                profit_loss = position_size * (
                    (execution_price - entry_price) / entry_price
                )
                # Update the balance
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

                # Update the balance
                self.closing_fee = position_size * self.limit_fee
                self.balance += (profit_loss - self.closing_fee - self.opening_fee) + (
                    position_size / previous_leverage
                )
                self.realized_pnl = profit_loss - self.closing_fee - self.opening_fee

        else:
            # Adapt the order if it is coming from a margin call

            if position_type == "long" and order_type == "sell":

                # Compute the loss
                realized_profit = position_size * (
                    (margin_price - entry_price) / entry_price
                )

                # Update the balance
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

            elif position_type == "short" and order_type == "buy":

                # Compute the loss
                realized_profit = position_size * (
                    (entry_price - margin_price) / entry_price
                )

                # Update the balance
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

            if (
                position_type == "long"
                and margin_price > worst_case_price
                and margin_price > stop_loss_price
            ):
                self.handle_margin_call(worst_case_price, margin_price)

            elif (
                position_type == "short"
                and margin_price < worst_case_price
                and margin_price < stop_loss_price
            ):
                self.handle_margin_call(worst_case_price, margin_price)

        else:

            if position_type == "long" and margin_price > worst_case_price:
                self.handle_margin_call(worst_case_price, margin_price)

            elif position_type == "short" and margin_price < worst_case_price:
                self.handle_margin_call(worst_case_price, margin_price)

    def handle_margin_call(self, worst_case_price, margin_price):

        # Mark that a margin call was triggered
        self.margin_call_triggered = True

        if not self.positions:
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

        max_dd = calculate_max_drawdown(self.history)

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
        self.sharpe_ratio = rolling_sharpe

        # # Full episode sharpe ratio

        # std_dev = np.std(returns) if len(returns) > 1 else 0
        # sharpe_ratio = (
        #     (np.mean(returns) - RISK_FREE_RATE) / std_dev if std_dev != 0 else 0
        # )

        # Add log-returns as reward
        log_returns = np.array(self.log_trading_returns)
        if len(log_returns) < 1 or np.isnan(log_returns).any():
            return 0  # Not enough data to calculate returns

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
            current_risk = min(current_risk, 1)
        else:
            current_risk = 0

        self.current_risk = current_risk

        # Adjust step return by current risk
        if current_risk > 0:
            risk_adjusted_step_return = log_returns[-1] / current_risk
        else:
            risk_adjusted_step_return = log_returns[-1]

        self.risk_adjusted_step = risk_adjusted_step_return

        penalty = 0

        if max_dd > self.previous_max_dd:
            # Penalize for increased drawdown
            penalty -= max_dd

        self.previous_max_dd = max_dd

        # Initialize compounded return
        self.compounded_returns *= 1 + returns[-1]

        # Calculate final compounded return
        final_compounded_return = self.compounded_returns - 1

        # # Activity reward
        # activity_reward = 0
        # if self.positions:
        #     activity_reward = 0.001

        # Only at the end of episode compounded return
        end_episode_comp_return = 0
        if (self.current_step + 1) >= (self.start_idx + self.episode_length):
            end_episode_comp_return = final_compounded_return

        # Reward leverage if returns are positive
        leverage_factor = 0.001 * self.previous_leverage
        leverage_bonus = 0
        if log_returns[-1] > leverage_factor:
            leverage_bonus = self.previous_leverage * 0.01

        # # Reward compounded returns if it is positive
        # compounded_return_bonus = 0
        # if final_compounded_return > 0:
        #     compounded_return_bonus = final_compounded_return

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
        # Plot the evolution of various metrics
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
