import numpy as np
import random
import time
from gurobipy import Model, GRB, quicksum
from matplotlib import pyplot as plt
from Instances.instance_8 import *  # ASSUMPTION: contains P and product-indexed data

ITER_MAX = 800
K_MAX = 3

# --------------------------
# 1) Greedy initial solution
# --------------------------
def generate_greedy_initial_solution(data):
    """Initial investment plan based on expected total demand across all products."""
    T, I, P, S, b, d, V, u, NB_T, NB_I = (
        data['T'], data['I'], data['P'], data['S'], data['b'], data['d'],
        data['V'], data['u'], data['NB_T'], data['NB_I']
    )

    # Use an ordered list for robust indexing
    T_list = sorted(list(T))
    tpos = {t: idx for idx, t in enumerate(T_list)}

    Y_initial = np.zeros((NB_T, NB_I), dtype=int)
    sites_used = [False] * NB_I

    # Expected total demand per period (sum over products, average over scenarios)
    avg_total_demand = {
        t: float(np.sum([np.mean([d[s, t, p] for s in S]) for p in P]))
        for t in T_list
    }

    for t in T_list:
        idx = tpos[t]
        cumulative_capacity = sum(b[i] * Y_initial[tp, i] for tp in range(idx + 1) for i in I)

        while cumulative_capacity < avg_total_demand[t]:
            best_site_to_build, best_ratio = -1, -1.0

            for i in I:
                if not sites_used[i]:
                    remaining_periods = NB_T - idx  # maintenance from install period to end
                    total_first_stage_cost = (V * b[i]) + (remaining_periods * u * b[i])

                    if total_first_stage_cost > 0:
                        ratio = b[i] / total_first_stage_cost
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_site_to_build = i

            if best_site_to_build != -1:
                i = best_site_to_build
                Y_initial[idx, i] = 1
                sites_used[i] = True
                cumulative_capacity += b[i]
            else:
                break

    return Y_initial


# --------------------------
# 2) Shake / Neighborhoods
# --------------------------
def shake(Y_current, k, data):
    """Neighborhood operator on Y (works on period-index rows 0..NB_T-1)."""
    T, I, NB_T = data['T'], data['I'], data['NB_T']
    T_list = sorted(list(T))  # only used for length consistency

    Y_shaken = Y_current.copy()
    built = list(zip(*np.where(Y_shaken == 1)))  # (t_index, i)

    if k == 1:
        if not built:
            return Y_shaken
        t_old, i_move = random.choice(built)
        available_times = [tt for tt in range(NB_T) if tt != t_old]
        if not available_times:
            return Y_shaken
        t_new = random.choice(available_times)
        Y_shaken[t_old, i_move], Y_shaken[t_new, i_move] = 0, 1
        return Y_shaken

    elif k == 2:
        if len(built) < 2:
            return shake(Y_current, 1, data)
        (t1, i1), (t2, i2) = random.sample(built, 2)
        Y_shaken[t1, i1], Y_shaken[t2, i2] = 0, 0
        Y_shaken[t2, i1], Y_shaken[t1, i2] = 1, 1
        return Y_shaken

    elif k == 3:
        sites_with_plants = {i for _, i in built}
        available_sites = [i for i in I if i not in sites_with_plants]

        can_add = len(available_sites) > 0
        can_remove = len(built) > 0

        if can_add and (not can_remove or random.random() < 0.5):
            t_add = random.randrange(NB_T)
            i_add = random.choice(available_sites)
            Y_shaken[t_add, i_add] = 1
        elif can_remove:
            t_rem, i_rem = random.choice(built)
            Y_shaken[t_rem, i_rem] = 0
        return Y_shaken

    else:
        # For k>3: compose multiple k=1 shakes
        Y_temp = Y_shaken
        for _ in range(k - 2):
            Y_temp = shake(Y_temp, 1, data)
        return Y_temp


# --------------------------
# 3) Recourse model (LP)
# --------------------------
def create_subproblem_model(data):
    """Creates the Gurobi second-stage subproblem structure (multi-product)."""
    sub_model = Model("SecondStage_MultiProduct")
    sub_model.setParam('OutputFlag', 0)
    sub_model.setParam("Method", 1)  # dual simplex for repeated RHS updates

    # IMPORTANT: keep these as per your edited preference
    sub_model.setParam("Presolve", 0)   # sometimes helps repeated RHS updates
    sub_model.setParam("Crossover", 0)  # if using simplex

    S, T, P = data['S'], data['T'], data['P']
    prob, pO, pC = data['prob'], data['pO'], data['pC']
    d, eO, eC = data['d'], data['eO'], data['eC']
    E_max, buy_price, sold_price = data['E_max'], data['buy_price'], data['sold_price']

    XO = sub_model.addVars(S, T, P, lb=0, name="XO")
    XC = sub_model.addVars(S, T, P, lb=0, name="XC")
    Buy = sub_model.addVars(S, T, lb=0, name="Buy")
    Sold = sub_model.addVars(S, T, lb=0, name="Sold")

    oper_cost = quicksum(
        prob[s] * (pO[s, t, p] * XO[s, t, p] + pC[s, t, p] * XC[s, t, p])
        for s in S for t in T for p in P
    )
    carbon_cost = quicksum(
        prob[s] * (buy_price[s, t] * Buy[s, t] - sold_price[s, t] * Sold[s, t])
        for s in S for t in T
    )

    sub_model.setObjective(oper_cost + carbon_cost, GRB.MINIMIZE)

    for s in S:
        for t in T:
            # Demand constraints per product
            for p in P:
                sub_model.addConstr(XO[s, t, p] + XC[s, t, p] == d[s, t, p],
                                    name=f"demand_{s}_{t}_{p}")

            # Emission cap with buy/sell variables
            sub_model.addConstr(
                quicksum(eO[s, t, p] * XO[s, t, p] + eC[s, t, p] * XC[s, t, p] for p in P)
                + Sold[s, t] <= E_max[t] + Buy[s, t],
                name=f"emission_cap_{s}_{t}"
            )

            # Prevent selling more than allocated (per period)
            sub_model.addConstr(Sold[s, t] <= E_max[t], name=f"emission_sold_and_max_{s}_{t}")

    # Capacity constraints RHS updated each evaluation
    capacity_constrs = {
        (s, t): sub_model.addConstr(quicksum(XO[s, t, p] for p in P) <= 0.0, name=f"cap_{s}_{t}")
        for s in S for t in T
    }

    sub_model.update()
    return sub_model, capacity_constrs


# --------------------------
# 4) Evaluate Y (update RHS)
# --------------------------
def evaluate_solution(Y, sub_model, capacity_constrs, data):
    """Evaluate total cost for a given investment matrix Y."""
    T, I, S = data['T'], data['I'], data['S']
    b, V, u = data['b'], data['V'], data['u']
    NB_T = data['NB_T']

    T_list = sorted(list(T))
    tpos = {t: idx for idx, t in enumerate(T_list)}

    # First-stage cost: install + maintenance from install period to end
    first_stage_cost = 0.0
    for t in T_list:
        idx = tpos[t]
        remaining_periods = NB_T - idx
        for i in I:
            if Y[idx, i] == 1:
                first_stage_cost += (V * b[i]) + (remaining_periods * u * b[i])

    # Cumulative capacity by period (in T's order)
    running = 0.0
    cumulative_capacity = {}
    for idx, t in enumerate(T_list):
        running += sum(b[i] * Y[idx, i] for i in I)
        cumulative_capacity[t] = running

    # Update capacity RHS
    for s in S:
        for t in T_list:
            capacity_constrs[(s, t)].RHS = float(cumulative_capacity[t])

    sub_model.optimize()
    if sub_model.status == GRB.OPTIMAL:
        return first_stage_cost + sub_model.ObjVal
    return float('inf')


# ============================================================
#        5) Q-learning: GRANULAR STATES + GRANULAR REWARD
# ============================================================
def compute_delta_norm(cost_best, cost_candidate):
    """Normalized improvement wrt best-so-far (positive is good)."""
    return (cost_best - cost_candidate) / max(abs(cost_best), 1.0)

def get_state_from_delta(delta_norm):
    """7-state discretization based on normalized improvement."""
    if delta_norm >= 0.02:
        return 0  # big improvement
    elif delta_norm >= 0.005:
        return 1  # medium improvement
    elif delta_norm > 0:
        return 2  # small improvement
    elif delta_norm >= -0.001:
        return 3  # neutral / tiny worsening
    elif delta_norm >= -0.005:
        return 4  # small worsening
    elif delta_norm >= -0.02:
        return 5  # medium worsening
    else:
        return 6  # big worsening

def get_reward_from_delta(delta_norm):
    """Granular reward; soft penalty for worsening to keep exploration."""
    return delta_norm if delta_norm > 0 else 0.2 * delta_norm

def initialize_q_table(states, actions):
    return {s: {a: 0.0 for a in actions} for s in states}

def choose_action(state, q_table, actions, epsilon):
    """Epsilon-greedy."""
    if random.random() < epsilon:
        return random.choice(actions)
    q_values = q_table[state]
    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q]
    return random.choice(best_actions)

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    max_next_q = max(q_table[next_state].values())
    old_value = q_table[state][action]
    q_table[state][action] = (1 - alpha) * old_value + alpha * (reward + gamma * max_next_q)


# --------------------------
# 6) Main QL-VNS procedure
# --------------------------
def QL_VNS(max_iterations, data, k_max=K_MAX,
           alpha=0.1, gamma=0.95, epsilon=0.9, epsilon_decay=0.999, epsilon_min=0.05):
    print("--- Starting QL-VNS (Granular States) for Multi-Product Problem ---")
    start_wall = time.time()

    # Q-learning setup
    states = list(range(7))               # 0..6 (granular)
    actions = list(range(1, k_max + 1))   # k=1..k_max
    q_table = initialize_q_table(states, actions)

    # Build recourse once
    sub_model, capacity_constrs = create_subproblem_model(data)

    # Initial solution
    # Y_best = np.zeros((NB_T, NB_I))
    Y_best = generate_greedy_initial_solution(data)
    cost_best = evaluate_solution(Y_best, sub_model, capacity_constrs, data)
    hist = [cost_best]

    print(f"Initial Solution Cost: {cost_best:.2f}\n")

    # Start neutral
    current_state = 3

    for it in range(max_iterations):
        # Choose neighborhood size
        action_k = choose_action(current_state, q_table, actions, epsilon)

        # Apply action
        Y_shaken = shake(Y_best, action_k, data)
        cost_shaken = evaluate_solution(Y_shaken, sub_model, capacity_constrs, data)

        # Compute granular delta vs best
        delta_norm = compute_delta_norm(cost_best, cost_shaken)
        reward = get_reward_from_delta(delta_norm)
        next_state = get_state_from_delta(delta_norm)

        # Accept only if improves best (keeps your original behavior)
        if cost_shaken < cost_best:
            cost_best = cost_shaken
            Y_best = Y_shaken
            print(f"Iter {it+1}: New best (k={action_k}) -> Cost: {cost_best:.2f}")

        # Update Q-table
        update_q_table(q_table, current_state, action_k, reward, next_state, alpha, gamma)
        

        # Transition
        current_state = next_state
        hist.append(cost_best)

        # Epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    end_wall = time.time()
    print(f"\n--- QL-VNS Finished in {end_wall - start_wall:.2f} seconds ---")
    return Y_best, cost_best, hist, q_table


# --------------------------
# 7) Execution
# --------------------------
if __name__ == "__main__":
    start_cpu = time.process_time()

    problem_data = {
        'T': T, 'I': I, 'S': S, 'P': P,
        'NB_T': NB_T, 'NB_I': NB_I, 'NB_S': NB_S, 'NB_P': NB_P,
        'b': b, 'd': d, 'V': V, 'u': u, 'prob': prob,
        'pO': pO, 'pC': pC,
        'buy_price': buy_price, 'sold_price': sold_price,
        'eO': eO, 'eC': eC, 'E_max': E_max
    }

    Y_opt, cost_opt, hist, q_table = QL_VNS(
        max_iterations=ITER_MAX,
        data=problem_data,
        k_max=K_MAX,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.9,
        epsilon_decay=0.999,
        epsilon_min=0.05
    )

    with open("Computational_analysis/collection_ql.txt", "a") as file:
        file.write(f"{np.round(cost_opt, 2)}\t{np.round(time.process_time() - start_cpu, 2)}\n")
    
    
    with open("Computational_analysis/iteration.py", "a") as interation:
        interation.write(f'hist_granular = {hist}\n')

    plt.plot(hist)
    plt.show()

    print("=====================================")