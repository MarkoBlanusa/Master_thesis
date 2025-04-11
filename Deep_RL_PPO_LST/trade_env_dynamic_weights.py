import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
import json


class TradingEnv(gym.Env):
    def __init__(
        self,
        data,
        input_length=5,
        market_fee=0.0005,
        limit_fee=0.0002,
        liquidation_fee=0.0125,
        slippage_mean=0.000001,
        slippage_std=0.00005,
        initial_balance=1000,
        total_episodes=1000,
        episode_length=192,  # 4 weeks of hourly data
        max_risk=0.02,
        min_risk=0.001,
        min_profit=0,
    ):
        super(TradingEnv, self).__init__()
        self.data = data
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
        self.technical_miss = 0.015
        self.no_trade = 0.001
        self.max_technical_miss = 0.15
        self.max_no_trade = 0.005
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

        # Define action space: weight, stop_loss, take_profit, leverage
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0, 0]),
            high=np.array([1, 1, 1, 1]),
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

        self.default_static_values = np.array(
            [
                self.unrealized_pnl,
                self.realized_pnl,
                self.compounded_returns,
                self.take_profit_price,
                self.stop_loss_price,
                self.entry_price,
                self.leverage,
                self.allowed_leverage,
                self.balance,
                self.position_value,
                self.desired_position_size,
                0,
                self.previous_max_dd,
                self.closing_fee,
                self.opening_fee,
                self.current_ask,
                self.current_bid,
                self.mark_price,
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

        # Randomly select a starting point in the dataset, ensuring there's enough data left for the episode
        self.start_idx = random.randint(0, len(self.data) - self.episode_length)
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
        self.action_history = []

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

        self.default_static_values = np.array(
            [
                self.unrealized_pnl,
                self.realized_pnl,
                self.compounded_returns,
                self.take_profit_price,
                self.stop_loss_price,
                self.entry_price,
                self.leverage,
                self.allowed_leverage,
                self.balance,
                self.position_value,
                self.desired_position_size,
                0,
                self.previous_max_dd,
                self.closing_fee,
                self.opening_fee,
                self.current_ask,
                self.current_bid,
                self.mark_price,
            ]
        )

        self.sequence_buffer.clear()
        self.load_initial_sequences()

        self.state = self.sequence_buffer[0]

        return self.state, {}

    def load_initial_sequences(self):
        """Initialize the sequence buffer based on the current episode starting index."""
        self.sequence_buffer.clear()

        # Load the initial sequence from the starting point in the dataset
        for i in range(self.input_length):
            initial_data_idx = self.start_idx + i
            if initial_data_idx < (self.start_idx + self.episode_length):
                initial_data = self.data[initial_data_idx]
                # Replace the last 18 columns with default static values
                initial_data[:, -18:] = self.default_static_values
                self.sequence_buffer.append(initial_data)
            else:
                break  # Stop if we've reached the episode length limit

    def update_sequences(self, new_data):
        """Update the sequence buffer as the episode progresses, adding new data and popping old data."""

        # Normalize the new data (as needed)
        normalized_new_data = new_data

        # Check if the next step exceeds the episode length
        if (self.current_step + self.input_length) < (
            self.start_idx + self.episode_length
        ):
            # Append the new sequence from the dataset if within episode bounds
            new_sequence_idx = self.current_step + self.input_length
            new_sequence = self.data[new_sequence_idx]
            self.sequence_buffer.append(new_sequence)
            self.sequence_buffer.pop(0)  # Remove the oldest sequence from the buffer
        else:
            self.sequence_buffer.pop(
                0
            )  # Only remove from the buffer if no new sequence is available

        # Update the last 18 columns of the sequences in the buffer with the new data
        if len(self.sequence_buffer) > 0:
            for i in range(len(self.sequence_buffer)):
                self.sequence_buffer[i][-i - 1, -18:] = normalized_new_data

            # Set the next state to be the first sequence in the buffer
            next_state = self.sequence_buffer[0]
            self.state = next_state
            self.current_step += 1
            terminated = self.current_step >= (self.start_idx + self.episode_length)

            return next_state, terminated, {}

        else:
            # If the buffer is empty, reset the environment (end of episode)
            next_state, info = self.reset()
            self.state = next_state
            terminated = True
            return next_state, terminated, info

    def get_std_dev_from_volume(
        self, volume, min_std=0.001, max_std=0.01, scaling_factor=7000
    ):
        # Calculate the inverse volume effect
        raw_std_dev = 1 / (volume / scaling_factor)

        # Normalize to a range between min_std and max_std
        normalized_std_dev = min_std + (max_std - min_std) * (
            raw_std_dev / (1 + raw_std_dev)
        )

        return normalized_std_dev

    def approximate_bid_ask(
        self,
        high_price,
        low_price,
        close_price,
        volume,
        bid_ask_std_base=0.001,
        scaling_factor=1000,
    ):

        range_price = high_price - low_price

        # Adjust std_dev based on volume
        std_dev = bid_ask_std_base / (volume / scaling_factor)

        bid_spread = np.random.normal(0, std_dev) * range_price
        ask_spread = np.random.normal(0, std_dev) * range_price

        bid_price = close_price - bid_spread
        ask_price = close_price + ask_spread

        return bid_price, ask_price

    def step(self, action):
        weight, stop_loss, take_profit, leverage = action
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

        # Get the latest observation from the current sequence for price logic
        current_price = self.state[-1, 3]
        current_high = self.state[-1, 1]
        current_low = self.state[-1, 2]
        current_volume = self.state[-1, 4]

        # Define a small tolerance value
        epsilon = 1e-10

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

        # Perform margin checks before attempting to open new positions
        self.check_margin_call(current_high, current_low)
        # Check for stop-loss, take-profit
        self.check_limits(current_high, current_low)

        # Total value of positions and current weight
        if self.positions:
            self.position_value = (sum(p[2] for p in self.positions)) / self.positions[
                -1
            ][3]
            current_weight = sum(
                [pos[2] * (1 if pos[0] == "long" else -1) for pos in self.positions]
            ) / ((self.balance + self.position_value) * self.positions[-1][3])

        else:
            self.position_value = 0
            current_weight = 0

        weight_diff = weight - current_weight

        # Calculate the desired position size
        desired_position_size = weight * (self.balance + self.position_value) * leverage
        self.desired_position_size = desired_position_size

        # Check the position size to see if it fits the margin rules and adapt the optimal leverage
        allowed_leverage, _, _, lower_size, upper_size = self.get_margin_tier(
            abs(desired_position_size)
        )
        self.allowed_leverage = allowed_leverage

        reward = 0

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

                # Update the unrealized pnl
                self.unrealized_pnl = (
                    self.positions[-1][2]
                    * ((rounded_bid - self.positions[-1][1]) / self.positions[-1][1])
                    if self.positions[-1][0] == "long"
                    else self.positions[-1][2]
                    * ((self.positions[-1][1] - rounded_ask) / self.positions[-1][1])
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

            # Calculate the difference in size based on the position type
            difference_size = abs(desired_position_size - current_position_size)
            desired_position_size = abs(desired_position_size)
            current_position_size = abs(current_position_size)

            # Compute the desired size in BTC
            desired_btc_size = desired_position_size / current_price

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

            if (
                desired_position_size >= self.min_size_usdt
                and desired_btc_size >= self.min_trade_btc_amount
                and desired_btc_size <= self.max_market_btc_amount
            ):

                if weight_diff > 0:  # Increase position
                    difference_size = min(difference_size, max_position_size)

                    # Increase the long position
                    if current_weight > 0:

                        increasing_size = difference_size
                        required_margin = increasing_size / self.leverage
                        position_type, entry_price, current_size, previous_leverage = (
                            self.positions[-1]
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

                            # Calculate and deduct the transaction fee
                            self.opening_fee += increasing_size * self.market_fee

                            # Check if having a stop loss under the risk level is feasible
                            max_loss_per_unit = (
                                self.max_risk * (self.balance + self.position_value)
                                - (2 * self.opening_fee)
                            ) / new_size
                            min_loss_per_unit = (
                                self.min_risk * (self.balance + self.position_value)
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
                                    (effective_market_bid_price - new_entry_price)
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
                                self.opening_fee -= increasing_size * self.market_fee

                                # Check if having a stop loss under the risk level is feasible
                                max_loss_per_unit = (
                                    self.max_risk * (self.balance + self.position_value)
                                    - (2 * self.opening_fee)
                                ) / current_size
                                min_loss_per_unit = (
                                    self.min_risk * (self.balance + self.position_value)
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

                            # Check if having a stop loss under the risk level is feasible
                            max_loss_per_unit = (
                                self.max_risk * (self.balance + self.position_value)
                                - (2 * self.opening_fee)
                            ) / current_size
                            min_loss_per_unit = (
                                self.min_risk * (self.balance + self.position_value)
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

                    # Diminishing the short position and potentitally closing it to open a long position
                    if current_weight < 0:  # Closing a short position

                        # Ensure to not close more than the current short position
                        closing_size = min(difference_size, current_position_size)

                        # If the entire position is not closed, update the remaining position
                        if closing_size < current_position_size:
                            # Check if we can afford to keep the remaining position while still respecting the risk tolerance

                            remaining_size = current_position_size - closing_size

                            # Execute the buy order to close part of the short position
                            realized_pnl = closing_size * (
                                (self.positions[0][1] - effective_market_ask_price)
                                / self.positions[0][1]
                            )

                            # Update the opening fee
                            self.opening_fee -= closing_size * self.market_fee

                            # Compute the closing fee
                            self.closing_fee = closing_size * self.market_fee

                            # Check if having a stop loss under the risk level is feasible
                            max_loss_per_unit = (
                                self.max_risk
                                * (
                                    (
                                        self.balance
                                        + realized_pnl
                                        - 2 * self.closing_fee
                                        + (closing_size / self.positions[-1][3])
                                    )
                                    + (remaining_size / self.positions[-1][3])
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
                                        + (closing_size / self.positions[-1][3])
                                    )
                                    + (remaining_size / self.positions[-1][3])
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

                                self.realized_pnl = realized_pnl - 2 * self.closing_fee

                                self.cumulative_transaction_costs += (
                                    closing_size * self.market_fee
                                )

                                # Update the unrealized pnl
                                self.unrealized_pnl = remaining_size * (
                                    (self.positions[0][1] - effective_market_ask_price)
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
                                self.opening_fee += closing_size * self.market_fee

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
                                    "short",
                                    adjusted_stop_loss,
                                    mark_price,
                                )

                                # Update the unrealized pnl
                                self.unrealized_pnl = self.positions[-1][2] * (
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
                        current_weight < 0
                        and (difference_size - current_position_size) > 0
                    ) or (
                        current_weight == 0
                        and (difference_size - current_position_size) > 0
                    ):
                        new_position_size = difference_size - current_position_size
                        required_margin = new_position_size / self.leverage

                        if self.balance >= (required_margin - epsilon):

                            # Calculate and deduct the transcation fee separately
                            self.opening_fee = new_position_size * self.market_fee

                            # Check if having a stop loss under the risk level is feasible
                            max_loss_per_unit = (
                                self.max_risk * (self.balance + self.position_value)
                                - (2 * self.opening_fee)
                            ) / new_position_size
                            min_loss_per_unit = (
                                self.min_risk * (self.balance + self.position_value)
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
                                self.opening_fee -= new_position_size * self.market_fee

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

                elif weight_diff < 0:  # Decrease position
                    difference_size = min(difference_size, max_position_size)

                    # Increase the short position
                    if current_weight < 0:
                        increasing_size = difference_size
                        required_margin = increasing_size / self.leverage
                        position_type, entry_price, current_size, previous_leverage = (
                            self.positions[-1]
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

                            # Calculate and deduct the transcation fee separately
                            self.opening_fee += increasing_size * self.market_fee

                            # Check if having a stop loss under the risk level is feasible
                            max_loss_per_unit = (
                                self.max_risk * (self.balance + self.position_value)
                                - (2 * self.opening_fee)
                            ) / new_size
                            min_loss_per_unit = (
                                self.min_risk * (self.balance + self.position_value)
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
                                    (new_entry_price - effective_market_ask_price)
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
                                self.opening_fee -= increasing_size * self.market_fee

                                # Check if having a stop loss under the risk level is feasible
                                max_loss_per_unit = (
                                    self.max_risk * (self.balance + self.position_value)
                                    - (2 * self.opening_fee)
                                ) / current_size
                                min_loss_per_unit = (
                                    self.min_risk * (self.balance + self.position_value)
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

                            # Check if having a stop loss under the risk level is feasible
                            max_loss_per_unit = (
                                self.max_risk * (self.balance + self.position_value)
                                - (2 * self.opening_fee)
                            ) / current_size
                            min_loss_per_unit = (
                                self.min_risk * (self.balance + self.position_value)
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

                    # Diminishing the long position and potentitally closing it to open a short position
                    if current_weight > 0:  # Closing a long position

                        # Ensure to not close more than the current long position
                        closing_size = min(difference_size, current_position_size)

                        # If the entire position is not closed, update the remaining position
                        if closing_size < current_position_size:

                            # Check if we can afford to keep the remaining position while still respecting the risk tolerance
                            remaining_size = current_position_size - closing_size

                            realized_pnl = closing_size * (
                                (effective_market_bid_price - self.positions[0][1])
                                / self.positions[0][1]
                            )

                            # Update the opening fee
                            self.opening_fee -= closing_size * self.market_fee

                            # Compute the closing fee
                            self.closing_fee = closing_size * self.market_fee

                            # Check if having a stop loss under the risk level is feasible
                            max_loss_per_unit = (
                                self.max_risk
                                * (
                                    (
                                        self.balance
                                        + realized_pnl
                                        - 2 * self.closing_fee
                                        + (closing_size / self.positions[-1][3])
                                    )
                                    + (remaining_size / self.positions[-1][3])
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
                                        + (closing_size / self.positions[-1][3])
                                    )
                                    + (remaining_size / self.positions[-1][3])
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

                                self.realized_pnl = realized_pnl - 2 * self.closing_fee

                                self.cumulative_transaction_costs += (
                                    closing_size * self.market_fee
                                )

                                # Update the unrealized pnl
                                self.unrealized_pnl = remaining_size * (
                                    (effective_market_bid_price - self.positions[0][1])
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
                                self.opening_fee += closing_size * self.market_fee

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
                                    "long",
                                    adjusted_stop_loss,
                                    mark_price,
                                )

                                # Update the unrealized pnl
                                self.unrealized_pnl = self.positions[-1][2] * (
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
                        current_weight > 0
                        and (difference_size - current_position_size) > 0
                    ) or (
                        current_weight == 0
                        and (difference_size - current_position_size) > 0
                    ):
                        new_position_size = difference_size - current_position_size
                        required_margin = new_position_size / self.leverage

                        if self.balance >= (required_margin - epsilon):

                            # Calculate and deduct the transcation fee separately
                            self.opening_fee = new_position_size * self.market_fee

                            # Check if having a stop loss under the risk level is feasible
                            max_loss_per_unit = (
                                self.max_risk * (self.balance + self.position_value)
                                - (2 * self.opening_fee)
                            ) / new_position_size
                            min_loss_per_unit = (
                                self.min_risk * (self.balance + self.position_value)
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
                                self.opening_fee -= new_position_size * self.market_fee

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

                else:
                    # No changes in the weight

                    # Check if there is a position open first
                    # Check if despite not changing the position, the new stop loss is compatible with the risk tolerance
                    if self.positions:

                        position_type, entry_price, position_size, previous_leverage = (
                            self.positions[-1]
                        )

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
                            (1 - self.cap_rate) * (mark_price / self.positions[-1][1])
                        )

                        # Compute the max
                        final_max_loss_per_unit = min(
                            max_loss_per_unit, restricted_loss_per_unit
                        )

                        adjusted_stop_loss = max(
                            min(stop_loss, final_max_loss_per_unit), min_loss_per_unit
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
                                realized_pnl - self.closing_fee - self.opening_fee
                            ) + (position_size / previous_leverage)

                            self.realized_pnl = (
                                realized_pnl - self.closing_fee - self.opening_fee
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
                # Minimum size and trade amount aren't met

                # Update the limit orders even if we keep the same position
                # Check if there is an open position
                if self.positions:

                    penalty = min(
                        self.technical_miss
                        + (0.5 * self.technical_miss * self.consecutive_technical_miss),
                        self.max_technical_miss,
                    )
                    reward -= penalty
                    self.consecutive_technical_miss += 1

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

                    # Set new limits even if the position is the same
                    self.set_limits(
                        self.positions[-1][1],
                        take_profit,
                        self.positions[-1][0],
                        adjusted_stop_loss,
                        mark_price,
                    )

                else:
                    # Wrong size while no positions
                    penalty = min(
                        self.technical_miss
                        + (0.5 * self.technical_miss * self.consecutive_technical_miss),
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
            self.stop_loss_price, _ = self.stop_loss_levels[self.entry_price]
            if self.take_profit_levels:
                self.take_profit_price, _ = self.take_profit_levels[self.entry_price]
            else:
                self.take_profit_price = 0
        else:
            self.position_value = 0
            self.entry_price = 0
            self.stop_loss_price = 0
            self.take_profit_price = 0

        # UPDATE THE PORTFOLIO VALUE
        self.portfolio_value = self.balance + self.position_value
        # print("PORTFOLIO VALUE : ", self.portfolio_value)
        self.history.append(self.portfolio_value)

        # Log trading return if no margin call was triggered
        if len(self.history) > 1:
            if not self.margin_call_triggered:
                trading_return = (self.history[-1] - self.history[-2]) / self.history[
                    -2
                ]
                self.trading_returns.append(trading_return)
                self.episode_metrics["returns"].append(trading_return)

                # Calculate log trading return
                if self.history[-1] == 0 or self.history[-2] == 0:
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

        # Initialize the 11 additional variables + realized PnL
        additional_state = np.array(
            [
                self.unrealized_pnl,
                self.realized_pnl,
                self.compounded_returns,
                self.take_profit_price,
                self.stop_loss_price,
                self.entry_price,
                self.leverage,
                self.allowed_leverage,
                self.balance,
                self.position_value,
                self.desired_position_size,
                self.log_trading_returns[-1],
                self.previous_max_dd,
                self.closing_fee,
                self.opening_fee,
                self.current_ask,
                self.current_bid,
                self.mark_price,
            ]
        )

        self.current_step += 1
        next_state, terminated, info = self.update_sequences(additional_state)
        self.state = next_state

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
                self.balance += (
                    realized_profit - self.closing_fee - self.opening_fee - margin_fee
                ) + (position_size / previous_leverage)
                self.realized_pnl = (
                    realized_profit - self.closing_fee - self.opening_fee - margin_fee
                )

            elif position_type == "short" and order_type == "buy":

                # Compute the loss
                realized_profit = position_size * (
                    (entry_price - margin_price) / entry_price
                )

                # Update the balance
                self.closing_fee = position_size * self.limit_fee
                margin_fee = position_size * self.liquidation_fee
                self.balance += (
                    realized_profit - self.closing_fee - self.opening_fee - margin_fee
                ) + (position_size / previous_leverage)
                self.realized_pnl = (
                    realized_profit - self.closing_fee - self.opening_fee - margin_fee
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

    def check_margin_call(self, high_price, low_price):

        if not self.positions:
            return

        # Calculate the total position value based on worst-case scenario
        worst_case_price = (
            low_price if any(p[0] == "long" for p in self.positions) else high_price
        )

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

        # Clear all stop loss and take profit levels
        self.stop_loss_levels.clear()
        self.take_profit_levels.clear()

        self.episode_metrics["num_margin_calls"] += 1  # Log the number of margin calls
        self.episode_metrics["list_margin_calls"].append(
            self.episode_metrics["num_margin_calls"]
        )

        # Log the trading return before resetting, without counting the reset as a return
        if len(self.history) > 0:
            trading_return = (self.portfolio_value - self.history[-1]) / self.history[
                -1
            ]
            self.trading_returns.append(trading_return)
            self.episode_metrics["returns"].append(trading_return)
            # Calculate log trading return
            if self.portfolio_value == 0 or self.history[-1] == 0:
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

        penalty = 0

        if max_dd > self.previous_max_dd:
            # Penalize for increased drawdown
            penalty -= max_dd

        self.previous_max_dd = max_dd

        # Check if there were margin calls and apply penalty to the reward
        if self.margin_call_triggered:
            margin_call_penalty = 0
        else:
            margin_call_penalty = 0

        # Add penalty for the number of stop-loss hits
        stop_loss_hits = self.episode_metrics.get("stop_loss_hits", 0)
        penalty -= stop_loss_hits * 0

        # Add log-returns as reward
        log_returns = np.array(self.log_trading_returns)
        if len(log_returns) < 1 or np.isnan(log_returns).any():
            return 0  # Not enough data to calculate returns

        # Initialize compounded return
        self.compounded_returns *= 1 + returns[-1]

        # Calculate final compounded return
        final_compounded_return = self.compounded_returns - 1

        reward = (
            (log_returns[-1] * 9)
            + (final_compounded_return * 0.1)
            + penalty
            + margin_call_penalty
        )

        return reward

    def log_step_metrics(self):

        # Log risk taken with stop loss levels
        if self.positions:
            position_type, entry_price, position_size, previous_leverage = (
                self.positions[-1]
            )
            stop_loss_price, _ = self.stop_loss_levels[entry_price]
            current_risk = (
                (position_size * ((entry_price - stop_loss_price) / entry_price))
                / self.portfolio_value
                if position_type == "long"
                else (position_size * ((stop_loss_price - entry_price) / entry_price))
                / self.portfolio_value
            )
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
        pass


# Example functions for risk measures (implement these as needed)
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
