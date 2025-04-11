import numpy as np
import matplotlib.pyplot as plt
import random

# Constants for penalties and reward factors
MAX_COMPOUNDED_RETURN_PENALTY = (
    -1
)  # Max compounded return penalty (2% of balance per trade)
LOG_RETURN_FACTOR = 9.0
COMPOUNDED_RETURN_FACTOR = 0.1

# Trade simulation constants
NUM_TRADES = 288  # Number of simulated trades
TRADE_PROBABILITY = 0.7  # Probability that the model decides to trade
MAX_TAKE_PROFIT = 0.05  # 5% max profit for take profit trades
RANDOM_PROFIT_LOSS = 0.1  # Random fluctuation of profit or loss

# Penalty ranges and increments
NO_TRADE_PENALTY_START = 0.001
NO_TRADE_PENALTY_MAX = 0.005
NO_TRADE_PENALTY_INCREMENT = 0.001 * 0.50  # 50% of the starting value

TECH_ENTRY_PENALTY_START = 0.015
TECH_ENTRY_PENALTY_MAX = 0.15
TECH_ENTRY_PENALTY_INCREMENT = 0.015 * 0.50  # 50% of the starting value


# Reward system simulation with dynamic penalties
def simulate_trade_system(
    log_factor=LOG_RETURN_FACTOR, comp_factor=COMPOUNDED_RETURN_FACTOR
):
    log_returns = []
    compounded_returns = []
    technical_entry_penalties = []
    non_entry_penalties = []
    rewards = []

    compounded_balance = 1.0  # Starting balance for compounded returns
    balance = 1000  # Starting balance for log return calculations

    no_trade_penalty = NO_TRADE_PENALTY_START
    tech_entry_penalty = TECH_ENTRY_PENALTY_START
    consecutive_no_trades = 0
    consecutive_missed_entries = 0

    for i in range(NUM_TRADES):
        # Decide if the model trades or not
        trade = np.random.rand() < TRADE_PROBABILITY

        if trade:
            # Simulate profit or loss outcome
            profit_loss = np.random.uniform(-RANDOM_PROFIT_LOSS, MAX_TAKE_PROFIT)
            log_return = profit_loss * log_factor

            # Cap the log return penalty
            if log_return < MAX_COMPOUNDED_RETURN_PENALTY:
                log_return = MAX_COMPOUNDED_RETURN_PENALTY

            # Update compounded returns
            compounded_balance *= 1 + profit_loss
            compounded_return = (compounded_balance - balance) / balance * comp_factor

            # Cap the compounded return penalty
            if compounded_return < MAX_COMPOUNDED_RETURN_PENALTY:
                compounded_return = MAX_COMPOUNDED_RETURN_PENALTY

            # Calculate total reward
            reward = log_return + compounded_return
            rewards.append(reward)

            # Log the returns
            log_returns.append(log_return)
            compounded_returns.append(compounded_return)

            # Reset penalties on a valid trade
            no_trade_penalty = NO_TRADE_PENALTY_START
            tech_entry_penalty = TECH_ENTRY_PENALTY_START
            consecutive_no_trades = 0
            consecutive_missed_entries = 0

        else:
            # Penalty for no trading (increasing penalty with consecutive misses)
            non_entry_penalty = no_trade_penalty
            no_trade_penalty = min(
                no_trade_penalty + NO_TRADE_PENALTY_INCREMENT, NO_TRADE_PENALTY_MAX
            )
            consecutive_no_trades += 1

            # Penalty for missed technical entries (increasing with consecutive misses)
            missed_entry_penalty = tech_entry_penalty
            tech_entry_penalty = min(
                tech_entry_penalty + TECH_ENTRY_PENALTY_INCREMENT,
                TECH_ENTRY_PENALTY_MAX,
            )
            consecutive_missed_entries += 1

            # Log penalties
            technical_entry_penalties.append(missed_entry_penalty)
            non_entry_penalties.append(non_entry_penalty)

            # Total penalty for not trading
            reward = -missed_entry_penalty - non_entry_penalty
            rewards.append(reward)

    return {
        "log_returns": np.array(log_returns),
        "compounded_returns": np.array(compounded_returns),
        "technical_entry_penalties": np.array(technical_entry_penalties),
        "non_entry_penalties": np.array(non_entry_penalties),
        "rewards": np.array(rewards),
    }


# Plotting function to visualize reward system results
def plot_rewards(result):
    rewards = result["rewards"]
    log_returns = result["log_returns"]
    compounded_returns = result["compounded_returns"]
    tech_entry_penalties = result["technical_entry_penalties"]
    non_entry_penalties = result["non_entry_penalties"]

    plt.figure(figsize=(12, 8))

    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(rewards, label="Rewards")
    plt.title("Total Rewards per Trade")
    plt.ylabel("Reward")
    plt.legend()

    # Plot log returns and compounded returns
    plt.subplot(3, 1, 2)
    plt.plot(log_returns, label="Log Returns")
    plt.plot(compounded_returns, label="Compounded Returns")
    plt.title("Log and Compounded Returns per Trade")
    plt.ylabel("Return")
    plt.legend()

    # Plot penalties
    plt.subplot(3, 1, 3)
    plt.plot(tech_entry_penalties, label="Technical Entry Penalties")
    plt.plot(non_entry_penalties, label="Non-Entry Penalties")
    plt.title("Penalties per Trade")
    plt.ylabel("Penalty")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Simulate and plot rewards with dynamic penalties
result = simulate_trade_system()
plot_rewards(result)

j = 0
for i in range(100):
    j += 1
    print(random.randint(0, 0))
    print(j)
