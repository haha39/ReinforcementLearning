import numpy as np
import matplotlib.pyplot as plt

# Parameter Settings
num_arms = 10
num_steps = 1000
num_runs = 100

# Creating a Bandit Environment
def init_bandit():
    q_star = np.random.normal(0, 1, num_arms)
    return q_star

# ε-greedy strategy
def epsilon_greedy(q_star, epsilon, decay=False):
    rewards = np.zeros(num_steps)
    counts = np.zeros(num_arms)
    estimates = np.zeros(num_arms)

    for t in range(num_steps):
        if decay:   # the fourth experiment 
            epsilon_t = max(0.01, epsilon * (0.99 ** t))
        else:
            epsilon_t = epsilon

        if np.random.rand() < epsilon_t:    # exploration
            action = np.random.randint(num_arms)
        else:                               # exploitation             
            action = np.argmax(estimates)

        reward = np.random.normal(q_star[action], 1)
        counts[action] += 1
        estimates[action] += (reward - estimates[action]) / counts[action]
        rewards[t] = reward

    return rewards

# Perform the experiment
def run_experiments(epsilon, decay=False):
    all_rewards = np.zeros(num_steps)
    for run in range(num_runs):
        q_star = init_bandit()
        rewards = epsilon_greedy(q_star, epsilon, decay)
        all_rewards += rewards
    return all_rewards / num_runs

# Drawing & Saving
plt.figure(figsize=(10,6))

for eps in [0, 0.01, 0.1]:
    avg_rewards = run_experiments(eps)
    plt.plot(avg_rewards, label=f"ε = {eps}")

avg_rewards_decay = run_experiments(1.0, decay=True)
plt.plot(avg_rewards_decay, label="ε decaying", linestyle='--')

plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("10-Armed Bandit: Average Reward over Time")
plt.legend()
plt.grid(True)

# Save Image
plt.savefig("average_reward_plot.png", dpi=300)
plt.show()

