import numpy as np
import random
import time
from collections import defaultdict
from gurobipy import Model, GRB, quicksum
from Instances.instance_10 import *
import matplotlib.pyplot as plt

# ============================================================
#                 QL-VNS (Q-learning + VNS Moves)
#   Keeps your structure: same greedy init, same shake(), same
#   Gurobi recourse evaluation with RHS updates.
# ============================================================

start_time = time.process_time()

# ----------------- User parameters -----------------
MAX_EPISODES = 60
STEPS_PER_EPISODE = 40
K_MAX = 3

# Q-learning hyperparameters
ALPHA = 0.2
GAMMA = 0.95
EPS_START = 0.8
EPS_MIN = 0.05
EPS_DECAY = 0.995

_values = []

# --- 1. Bundle Data for Cleanliness and Portability ---
problem_data = {
    'T': T, 'I': I, 'S': S, 'P': P,
    'NB_T': NB_T, 'NB_I': NB_I, 'NB_S': NB_S, 'NB_P': NB_P,
    'b': b, 'd': d, 'V': V, 'u': u, 'prob': prob,
    'pO': pO, 'pC': pC, 'buy_price': buy_price, 'sold_price': sold_price,
    'eO': eO, 'eC': eC, 'E_max': E_max
}

# ------------------------------------------------------------
# 2. Greedy initial solution (multi-product aggregate demand)
# ------------------------------------------------------------
def generate_greedy_initial_solution(data):
    """Generates an initial investment plan based on total aggregate demand."""
    T, I, P, S, b, d, V, u, NB_T, NB_I = (
        data['T'], data['I'], data['P'], data['S'], data['b'], data['d'],
        data['V'], data['u'], data['NB_T'], data['NB_I']
    )

    Y_initial = np.zeros((NB_T, NB_I), dtype=int)
    sites_used = [False] * NB_I

    # Average TOTAL demand across all products for each period
    avg_total_demand = {
        t: np.sum([np.mean([d[s, t, j] for s in S]) for j in P])
        for t in T
    }

    for t in T:
        cumulative_capacity = sum(b[i] * Y_initial[tp, i] for tp in range(t + 1) for i in I)

        while cumulative_capacity < avg_total_demand[t]:
            best_site_to_build, max_benefit_cost_ratio = -1, -1
            for i in I:
                if not sites_used[i]:
                    install_cost = V * b[i]
                    maintenance_cost = (NB_T - t) * u * b[i]
                    total_first_stage_cost = install_cost + maintenance_cost
                    if total_first_stage_cost > 0:
                        ratio = b[i] / total_first_stage_cost
                        if ratio > max_benefit_cost_ratio:
                            max_benefit_cost_ratio = ratio
                            best_site_to_build = i

            if best_site_to_build != -1:
                i = best_site_to_build
                Y_initial[t, i] = 1
                sites_used[i] = True
                cumulative_capacity += b[i]
            else:
                break
    return Y_initial

# ------------------------------------------------------------
# 3. Shake (neighborhood moves) -- unchanged
# ------------------------------------------------------------
def shake(Y_current, k, data):
    T, I = data['T'], data['I']
    Y_shaken = Y_current.copy()
    built_plants = list(zip(*np.where(Y_shaken == 1)))

    if k == 1:
        if not built_plants:
            return Y_shaken
        t_old, i_to_move = random.choice(built_plants)
        available_times = [t for t in T if t != t_old]
        if not available_times:
            return Y_shaken
        t_new = random.choice(available_times)
        Y_shaken[t_old, i_to_move], Y_shaken[t_new, i_to_move] = 0, 1

    elif k == 2:
        if len(built_plants) < 2:
            return shake(Y_current, 1, data)
        (t1, i1), (t2, i2) = random.sample(built_plants, 2)
        Y_shaken[t1, i1], Y_shaken[t2, i2] = 0, 0
        Y_shaken[t2, i1], Y_shaken[t1, i2] = 1, 1

    elif k == 3:
        sites_with_plants = {i for _, i in built_plants}
        available_sites = [i for i in I if i not in sites_with_plants]
        can_add = len(available_sites) > 0
        can_remove = len(built_plants) > 0
        if can_add and (not can_remove or random.random() < 0.5):
            Y_shaken[random.choice(T), random.choice(available_sites)] = 1
        elif can_remove:
            t_rem, i_rem = random.choice(built_plants)
            Y_shaken[t_rem, i_rem] = 0

    else:
        Y_temp = Y_shaken
        for _ in range(k - 2):
            Y_temp = shake(Y_temp, 1, data)
        return Y_temp

    return Y_shaken

# ------------------------------------------------------------
# 4. Second-stage subproblem model (multi-product)
# ------------------------------------------------------------
def create_subproblem_model(data):
    """Creates the Gurobi subproblem model structure for multiple products."""
    sub_model = Model("SecondStageSubproblem_MultiProduct")
    sub_model.setParam('OutputFlag', 0)
    sub_model.setParam("Method", 1)  # dual simplex good for repeated RHS updates

    # Edited as requested earlier:
    sub_model.setParam("Presolve", 0)  # sometimes helps repeated RHS updates
    sub_model.setParam("Crossover", 0) # if using simplex

    S, T, P, prob, pO, pC, buy_price, sold_price, d, eO, eC, E_max = (
        data['S'], data['T'], data['P'], data['prob'], data['pO'], data['pC'],
        data['buy_price'], data['sold_price'], data['d'], data['eO'], data['eC'], data['E_max']
    )

    XO = sub_model.addVars(S, T, P, lb=0, name="XO")
    XC = sub_model.addVars(S, T, P, lb=0, name="XC")
    Buy = sub_model.addVars(S, T, lb=0, name="Buy")
    Sold = sub_model.addVars(S, T, lb=0, name="Sold")

    oper_cost = quicksum(
        prob[s] * (pO[s, t, j] * XO[s, t, j] + pC[s, t, j] * XC[s, t, j])
        for s in S for t in T for j in P
    )
    carbon_cost = quicksum(
        prob[s] * (buy_price[s, t] * Buy[s, t] - sold_price[s, t] * Sold[s, t])
        for s in S for t in T
    )
    sub_model.setObjective(oper_cost + carbon_cost, GRB.MINIMIZE)

    for s in S:
        for t in T:
            # Demand satisfaction per product
            for j in P:
                sub_model.addConstr(XO[s, t, j] + XC[s, t, j] == d[s, t, j], name=f"demand_{s}_{t}_{j}")

            # Emissions summed over all products
            sub_model.addConstr(
                quicksum(eO[s, t, j] * XO[s, t, j] + eC[s, t, j] * XC[s, t, j] for j in P)
                + Sold[s, t] <= E_max[t] + Buy[s, t],
                name=f"emission_cap_{s}_{t}"
            )
            sub_model.addConstr(Sold[s, t] <= E_max[t], name=f"emission_sold_and_max_{s}_{t}")

    # Placeholder capacity constraints, RHS updated in evaluate_solution()
    capacity_constrs = {(s, t): sub_model.addConstr(quicksum(XO[s, t, j] for j in P) <= 0,
                                                    name=f"cap_{s}_{t}")
                        for s in S for t in T}

    sub_model.update()
    return sub_model, capacity_constrs

def evaluate_solution(Y, sub_model, capacity_constrs, data):
    """Evaluates a given Y for the multi-product problem."""
    T, I, b, V, u, NB_T, S = data['T'], data['I'], data['b'], data['V'], data['u'], data['NB_T'], data['S']

    # First-stage cost (unchanged from your implementation)
    first_stage_cost = (
        sum(V * b[i] * Y[t, i] for t in T for i in I)
        + sum((NB_T - t) * u * b[i] * Y[t, i] for t in T for i in I)
    )

    cumulative_capacity = {t: sum(b[i] * Y[tp, i] for tp in range(t + 1) for i in I) for t in T}

    for s in S:
        for t in T:
            capacity_constrs[(s, t)].RHS = float(cumulative_capacity[t])

    sub_model.optimize()

    if sub_model.status == GRB.OPTIMAL:
        return first_stage_cost + sub_model.ObjVal
    return float('inf')

# ------------------------------------------------------------
# 5. Q-learning utilities
# ------------------------------------------------------------
def extract_features(Y, cost, cost_best, data):
    """
    Compact state representation (tabular Q-learning needs small state space).
    Features:
      - gap bucket (cost vs best)
      - number of installations
      - average installation time index
    """
    NB_T, NB_I = data['NB_T'], data['NB_I']

    n_installed = int(np.sum(Y))

    pos = np.argwhere(Y == 1)
    if len(pos) == 0:
        avg_t = 0.0
    else:
        avg_t = float(np.mean(pos[:, 0]))

    if cost_best <= 1e-9:
        gap = 0.0
    else:
        gap = (cost - cost_best) / max(abs(cost_best), 1.0)

    gap_bucket = int(np.clip(np.floor(gap / 0.02), -50, 200))  # 2% buckets
    n_bucket = int(np.clip(n_installed, 0, NB_I))
    t_bucket = int(np.clip(np.floor(avg_t), 0, NB_T - 1))

    return (gap_bucket, n_bucket, t_bucket)

def choose_action(Q, state, actions, epsilon):
    """Epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.choice(actions)

    qvals = [Q[(state, a)] for a in actions]
    max_q = max(qvals)
    best = [a for a, q in zip(actions, qvals) if q == max_q]
    return random.choice(best)

# ------------------------------------------------------------
# 6. QL-VNS main loop
# ------------------------------------------------------------
def QL_VNS(data,
          k_max=3,
          max_episodes=60,
          steps_per_episode=40,
          alpha=0.2,
          gamma=0.95,
          epsilon_start=0.8,
          epsilon_min=0.05,
          epsilon_decay=0.995):
    """
    Q-learning selects which VNS neighborhood k to apply.
    Environment dynamics:
      Y_next = shake(Y_current, k)
      cost_next = evaluate_solution(Y_next)
    Reward:
      improvement = cost_current - cost_next (positive is good)
    """
    print("--- Starting QL-VNS for Multi-Product Problem ---")
    wall_start = time.time()

    sub_model, capacity_constrs = create_subproblem_model(data)

    print("Generating initial solution...")
    Y_best = generate_greedy_initial_solution(data)
    cost_best = evaluate_solution(Y_best, sub_model, capacity_constrs, data)
    print(f"Initial Solution Cost: {cost_best:.2f}\n")

    Y_current = Y_best.copy()
    cost_current = cost_best

    actions = list(range(1, k_max + 1))
    Q = defaultdict(float)
    epsilon = epsilon_start

    for ep in range(max_episodes):
        # restart each episode from best-found solution (stable + effective)
        Y_current = Y_best.copy()
        cost_current = cost_best

        for step in range(steps_per_episode):
            state = extract_features(Y_current, cost_current, cost_best, data)
            k = choose_action(Q, state, actions, epsilon)

            Y_next = shake(Y_current, k, data)
            cost_next = evaluate_solution(Y_next, sub_model, capacity_constrs, data)

            reward = cost_current - cost_next  # improvement

            # Update global best
            if cost_next < cost_best:
                Y_best = Y_next
                cost_best = cost_next
                print(f"Episode {ep}, step {step}: New best (k={k}) -> Cost: {cost_best:.2f}")

            next_state = extract_features(Y_next, cost_next, cost_best, data)
            best_next_Q = max(Q[(next_state, a)] for a in actions)

            Q[(state, k)] = Q[(state, k)] + alpha * (reward + gamma * best_next_Q - Q[(state, k)])

            Y_current = Y_next
            cost_current = cost_next

            _values.append(cost_best)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    wall_end = time.time()
    print(f"\n--- QL-VNS Finished in {wall_end - wall_start:.2f} seconds ---")
    return Y_best, cost_best, Q

# ------------------------------------------------------------
# 7. Execute
# ------------------------------------------------------------
if __name__ == "__main__":
    Y_opt, cost_opt, Q = QL_VNS(
        data=problem_data,
        k_max=K_MAX,
        max_episodes=MAX_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPS_START,
        epsilon_min=EPS_MIN,
        epsilon_decay=EPS_DECAY
    )

    # Save results (kept same filename to not break your pipeline)
    with open("Computational_analysis/collection_vns.txt", "a") as file:
        file.write(f"{np.round(cost_opt,2)} \t {np.round(time.process_time()-start_time,2)}\n")

    # Plot progress
    plt.plot(_values)
    plt.show()