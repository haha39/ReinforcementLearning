import numpy as np
import matplotlib.pyplot as plt

# GridWorld config
grid_size = 4
states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
actions = ['U', 'D', 'L', 'R']
action_map = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

# parameters
threshold = 1e-4
discount = 1.0

# terminal states
def is_terminal(state):
    return state == (0, 0) or state == (grid_size - 1, grid_size - 1)

def step(state, action):
    if is_terminal(state):
        return state, 0
    delta = action_map[action]
    next_state = (state[0] + delta[0], state[1] + delta[1])
    if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
        return next_state, -1
    else:
        return state, -1

def policy_evaluation(policy, V, k=None):
    iteration = 0
    while True:
        delta = 0
        V_new = V.copy()
        for s in states:
            if is_terminal(s):
                continue
            v = 0
            for a, prob in policy[s].items():
                next_state, reward = step(s, a)
                v += prob * (reward + discount * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V_new[s] = v
        V = V_new
        iteration += 1
        if k is not None and iteration >= k:
            break
        if delta < threshold:
            break
    return V, iteration

def policy_improvement(V):
    policy_stable = True
    new_policy = {}
    for s in states:
        if is_terminal(s):
            new_policy[s] = {}
            continue
        action_values = {}
        for a in actions:
            next_state, reward = step(s, a)
            action_values[a] = reward + discount * V[next_state]
        best_action = max(action_values, key=action_values.get)
        best_value = action_values[best_action]
        best_actions = [a for a, v in action_values.items() if v == best_value]
        new_policy[s] = {a: 1.0 / len(best_actions) for a in best_actions}
    return new_policy

# 1. Random policy
def random_policy():
    return {s: {a: 1/4 for a in actions} if not is_terminal(s) else {} for s in states}

# 2. Policy iteration
V = {s: 0 for s in states}
policy = random_policy()

# part 1: random policy at k=100
V_k100, _ = policy_evaluation(policy, V.copy(), k=100)

# Save state value to txt
with open("state_value_k100.txt", "w") as f:
    for i in range(grid_size):
        f.write(' '.join(f"{V_k100[(i,j)]:6.2f}" for j in range(grid_size)) + "\n")

# part 2: policy iteration
iteration_count = 0
while True:
    V, _ = policy_evaluation(policy, V)
    new_policy = policy_improvement(V)
    iteration_count += 1
    if new_policy == policy:
        break
    policy = new_policy

# Save optimal policy to txt
with open("optimal_policy.txt", "w") as f:
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            s = (i, j)
            if is_terminal(s):
                row.append(" T ")
            else:
                actions_str = ''.join(policy[s].keys())
                row.append(f"{actions_str:^3}")
        f.write(' '.join(row) + "\n")

# Visualization (optional)
def plot_state_values(V, title):
    grid = np.array([[V[(i, j)] for j in range(grid_size)] for i in range(grid_size)])
    plt.imshow(grid, cmap='coolwarm', interpolation='none')
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f"{grid[i, j]:.1f}", ha='center', va='center', color='black')
    plt.title(title)
    plt.colorbar()
    plt.savefig("state_value_k100.png")
    plt.show()

plot_state_values(V_k100, "State Value at k=100 (Random Policy)")

print("Optimal policy reached at iteration:", iteration_count)  # For 2.b
