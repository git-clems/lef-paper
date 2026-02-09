# import numpy as np
# import random
# import time
# from gurobipy import Model, GRB, quicksum
# from matplotlib import pyplot as plt
# from Instances.instance_7 import * # ASSUMPTION: This file contains P, and product-indexed data

# ITER_MAX = 350
# K_MAX = 3

# def generate_greedy_initial_solution(data):
#     """Generates an initial investment plan based on total aggregate demand across all products."""
#     T, I, P, S, b, d, V, u, NB_T, NB_I = (
#         data['T'], data['I'], data['P'], data['S'], data['b'], data['d'],
#         data['V'], data['u'], data['NB_T'], data['NB_I']
#     )
#     Y_initial = np.zeros((NB_T, NB_I), dtype=int)
#     sites_used = [False] * NB_I
#     avg_total_demand = {
#         t: np.sum([np.mean([d[s, t, p] for s in S]) for p in P])
#         for t in T
#     }
#     for t in T:
#         cumulative_capacity = sum(b[i] * Y_initial[tp, i] for tp in range(t + 1) for i in I)
#         while cumulative_capacity < avg_total_demand[t]:
#             best_site_to_build, max_benefit_cost_ratio = -1, -1
#             for i in I:
#                 if not sites_used[i]:
#                     total_first_stage_cost = (V * b[i]) + ((NB_T - t) * u * b[i])
#                     if total_first_stage_cost > 0:
#                         ratio = b[i] / total_first_stage_cost
#                         if ratio > max_benefit_cost_ratio:
#                             max_benefit_cost_ratio = ratio
#                             best_site_to_build = i
#             if best_site_to_build != -1:
#                 i = best_site_to_build
#                 Y_initial[t, i] = 1
#                 sites_used[i] = True
#                 cumulative_capacity += b[i]
#             else:
#                 break
#     return Y_initial

# def shake(Y_current, k, data):
#     """Shake function with corrected control flow."""
#     T, I = data['T'], data['I']
#     Y_shaken = Y_current.copy()
#     built_plants = list(zip(*np.where(Y_shaken == 1)))
#     if k == 1:
#         if not built_plants: return Y_shaken
#         t_old, i_to_move = random.choice(built_plants)
#         available_times = [t for t in T if t != t_old]
#         if not available_times: return Y_shaken
#         t_new = random.choice(available_times)
#         Y_shaken[t_old, i_to_move], Y_shaken[t_new, i_to_move] = 0, 1
#         return Y_shaken
#     elif k == 2:
#         if len(built_plants) < 2: return shake(Y_current, 1, data)
#         (t1, i1), (t2, i2) = random.sample(built_plants, 2)
#         Y_shaken[t1, i1], Y_shaken[t2, i2] = 0, 0
#         Y_shaken[t2, i1], Y_shaken[t1, i2] = 1, 1
#         return Y_shaken
#     elif k == 3:
#         sites_with_plants = {i for _, i in built_plants}
#         available_sites = [i for i in I if i not in sites_with_plants]
#         can_add = len(available_sites) > 0
#         can_remove = len(built_plants) > 0
#         if can_add and (not can_remove or random.random() < 0.5):
#             Y_shaken[random.choice(T), random.choice(available_sites)] = 1
#         elif can_remove:
#             t_rem, i_rem = random.choice(built_plants)
#             Y_shaken[t_rem, i_rem] = 0
#         return Y_shaken
#     else: # k > 3
#         Y_temp = Y_shaken
#         for _ in range(k - 2):
#             Y_temp = shake(Y_temp, 1, data)
#         return Y_temp

# def create_subproblem_model(data):
#     """Creates the Gurobi subproblem for multiple products."""
#     sub_model = Model("SecondStage_MultiProduct")
#     sub_model.setParam('OutputFlag', 0)
#     S, T, P, prob, pO, pC, d, eO, eC, E_max, buy_price, sold_price = (
#         data['S'], data['T'], data['P'], data['prob'], data['pO'], data['pC'], data['d'],
#         data['eO'], data['eC'], data['E_max'], data['buy_price'], data['sold_price']
#     )
#     XO, XC = sub_model.addVars(S, T, P, lb=0), sub_model.addVars(S, T, P, lb=0)
#     Buy, Sold = sub_model.addVars(S, T, lb=0), sub_model.addVars(S, T, lb=0)
#     oper_cost = quicksum(prob[s]*(pO[s,t,p]*XO[s,t,p] + pC[s,t,p]*XC[s,t,p]) for s in S for t in T for p in P)
#     carbon_cost = quicksum(prob[s]*(buy_price[s,t]*Buy[s,t] - sold_price[s,t]*Sold[s,t]) for s in S for t in T)
#     sub_model.setObjective(oper_cost + carbon_cost, GRB.MINIMIZE)
#     for s in S:
#         for t in T:
#             for p in P:
#                 sub_model.addConstr(XO[s,t,p] + XC[s,t,p] == d[s, t, p])
#             sub_model.addConstr(quicksum(eO[s,t,p]*XO[s,t,p] + eC[s,t,p]*XC[s,t,p] for p in P) + Sold[s,t] <= E_max[t] + Buy[s,t])
#             sub_model.addConstr(Sold[s,t] <= E_max[t], name="emission_sold_and_max_{s}_{t}")
            
#     capacity_constrs = {(s,t): sub_model.addConstr(quicksum(XO[s,t,p] for p in P) <= 0) for s in S for t in T}
#     sub_model.update()
#     return sub_model, capacity_constrs

# def evaluate_solution(Y, sub_model, capacity_constrs, data):
#     """Evaluates a given Y for the multi-product problem."""
#     T, I, S, b, V, u, NB_T = data['T'], data['I'], data['S'], data['b'], data['V'], data['u'], data['NB_T']
#     first_stage_cost = sum(V*b[i]*Y[t,i] for t in T for i in I) + \
#                        sum((NB_T - t)*u*b[i]*Y[t,i] for t in T for i in I)
#     cumulative_capacity = {t: sum(b[i] * Y[tp,i] for tp in range(t+1) for i in I) for t in T}
#     for s in S:
#         for t in T:
#             capacity_constrs[s, t].RHS = cumulative_capacity[t]
#     sub_model.optimize()
#     return first_stage_cost + sub_model.ObjVal if sub_model.status == GRB.OPTIMAL else float('inf')


# # --- 2. HELPER FUNCTIONS (FOR Q-LEARNING) ---

# def initialize_q_table(states, actions):
#     """Initializes the Q-table with zeros."""
#     return {s: {a: 0.0 for a in actions} for s in states}

# def choose_action(state, q_table, actions, epsilon):
#     """Chooses an action using an epsilon-greedy policy."""
#     if random.uniform(0, 1) < epsilon:
#         return random.choice(actions)
#     else:
#         q_values = q_table[state]
#         max_q = max(q_values.values())
#         best_actions = [a for a, q in q_values.items() if q == max_q]
#         return random.choice(best_actions)

# def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
#     """Updates the Q-value for a state-action pair using the Bellman equation."""
#     max_next_q = max(q_table[next_state].values())
#     learned_value = reward + gamma * max_next_q
#     old_value = q_table[state][action]
#     q_table[state][action] = (1 - alpha) * old_value + alpha * learned_value


# # --- 3. MAIN Q-LEARNING VNS PROCEDURE ---

# def Q_VNS(max_iterations, data, k_max=K_MAX, alpha=0.1, gamma=0.95, epsilon=0.9):
#     """Performs a VNS search guided by a Q-Learning agent."""
#     print("--- Starting Q-Learning VNS for Multi-Product Problem ---")
#     start_vns_time = time.time()
    

#     # --- Q-Learning Setup ---
#     states = [0, 1]  # 0: Improvement, 1: Stagnation
#     actions = list(range(1, k_max + 1)) # Corresponds to k=1, 2, ...
#     q_table = initialize_q_table(states, actions)

#     # --- VNS Initialization ---
#     sub_model, capacity_constrs = create_subproblem_model(data)
#     Y_best = generate_greedy_initial_solution(data)
#     cost_best = evaluate_solution(Y_best, sub_model, capacity_constrs, data)
#     hist = [cost_best]
#     print(f"Initial Solution Cost: {cost_best:.2f}\n")

#     current_state = 0 # Start in the "Improvement" state
#     for i in range(max_iterations):
#         # 1. Agent chooses an action (which neighborhood k to use)
#         action_k = choose_action(current_state, q_table, actions, epsilon)

#         # 2. Perform the action (shake the solution)
#         Y_shaken = shake(Y_best, action_k, data)
#         cost_shaken = evaluate_solution(Y_shaken, sub_model, capacity_constrs, data)

#         # 3. Determine the reward and the next state
#         if cost_shaken < cost_best:
#             reward = cost_best - cost_shaken
#             next_state = 0 # "Improvement" state
#             Y_best, cost_best = Y_shaken, cost_shaken # Update the best solution
#             # if (i + 1) % 10 == 0 or i==0:
#             print(f"Iter {i+1}: New best found (k={action_k}) -> Cost: {cost_best:.2f}")
#         else:
#             reward = -5.0
#             next_state = 1 # "Stagnation" 
        
#         hist.append(cost_best)

#         # 4. Update the Q-table with the experience
#         update_q_table(q_table, current_state, action_k, reward, next_state, alpha, gamma)

#         # 5. Transition to the next state
#         current_state = next_state

#     end_vns_time = time.time()
#     time.sleep(0.0)
#     print(f"\n--- Q-VNS Finished in {end_vns_time - start_vns_time:.2f} seconds ---")

#     # print("\nLearned Q-Table:")
#     # for s in states:
#     #     state_name = "Improvement" if s == 0 else "Stagnation"
#     #     print(f"  State: {state_name}")
#         # for a in actions:
#         #     print(f"    Action k={a}: Q-value = {q_table[s][a]:.3f}")

#     return Y_best, cost_best, hist


# # --- 4. EXECUTION BLOCK ---

# if __name__ == "__main__":
#     # Load multi-product data
#     start_time = time.process_time()
#     problem_data = {
#         'T': T, 'I': I, 'S': S, 'P': P, 'NB_T': NB_T, 'NB_I': NB_I, 'NB_S': NB_S, 'NB_P': NB_P,
#         'b': b, 'd': d, 'V': V, 'u': u, 'prob': prob, 'pO': pO, 'pC': pC,
#         'buy_price': buy_price, 'sold_price': sold_price, 'eO': eO, 'eC': eC, 'E_max': E_max
#     }
    

#     Y_opt, cost_opt, hist = Q_VNS(max_iterations=ITER_MAX, data=problem_data)
    
#     with open(f"Computational_analysis/collection_ql.txt", "a") as file:
#         file.write(f"{np.round(cost_opt,2)} \t {np.round(time.process_time()-start_time,2)}\n")
        
#     plt.plot(hist)
#     # plt.show()

#     # print("\n=====================================")
#     # print("           Final Results")
#     # print("=====================================")
#     # print("\nOptimal Investment Plan (Y):")
#     # print(Y_opt)
#     # print(f"Total process time : {np.round(time.process_time() - start_time,2)} second")
#     # print(f"\nOptimal Cost Found: {cost_opt:.2f}")
        
#     print("=====================================")

import numpy as np
import random
import time
from gurobipy import Model, GRB, quicksum
from matplotlib import pyplot as plt
from Instances.instance_12 import *  # assumes P and product-indexed data exist

ITER_MAX = 730
K_MAX = 3

# --------------------------
# 1) Greedy initial solution
# --------------------------
def generate_greedy_initial_solution(data):
    """Generates an initial investment plan based on total aggregate demand across all products."""
    T, I, P, S, b, d, V, u, NB_T, NB_I = (
        data['T'], data['I'], data['P'], data['S'], data['b'], data['d'],
        data['V'], data['u'], data['NB_T'], data['NB_I']
    )

    # Ensure we have a stable order for periods (works for {0..} or {1..})
    T_list = list(T)
    T_list.sort()

    Y_initial = np.zeros((NB_T, NB_I), dtype=int)
    sites_used = [False] * NB_I

    avg_total_demand = {
        t: float(np.sum([np.mean([d[s, t, p] for s in S]) for p in P]))
        for t in T_list
    }

    for t in T_list:
        # cumulative capacity up to t (using positions, not range(t+1))
        tpos = T_list.index(t)
        cumulative_capacity = sum(
            b[i] * Y_initial[tp, i]
            for tp in range(tpos + 1)
            for i in I
        )

        while cumulative_capacity < avg_total_demand[t]:
            best_site_to_build, max_ratio = -1, -1.0
            for i in I:
                if not sites_used[i]:
                    # maintenance from installation period to end (position-based)
                    remaining_periods = NB_T - tpos
                    total_first_stage_cost = (V * b[i]) + (remaining_periods * u * b[i])
                    if total_first_stage_cost > 0:
                        ratio = b[i] / total_first_stage_cost
                        if ratio > max_ratio:
                            max_ratio = ratio
                            best_site_to_build = i

            if best_site_to_build != -1:
                i = best_site_to_build
                # install at the row index corresponding to this period position
                Y_initial[tpos, i] = 1
                sites_used[i] = True
                cumulative_capacity += b[i]
            else:
                break

    return Y_initial


# --------------------------
# 2) Shake (unchanged logic)
# --------------------------
def shake(Y_current, k, data):
    """Shake function with corrected control flow."""
    T, I = data['T'], data['I']
    T_list = list(T); T_list.sort()

    Y_shaken = Y_current.copy()
    built_plants = list(zip(*np.where(Y_shaken == 1)))  # (t_index, i)

    if k == 1:
        if not built_plants:
            return Y_shaken
        t_old, i_to_move = random.choice(built_plants)
        available_times = [t for t in range(len(T_list)) if t != t_old]
        if not available_times:
            return Y_shaken
        t_new = random.choice(available_times)
        Y_shaken[t_old, i_to_move], Y_shaken[t_new, i_to_move] = 0, 1
        return Y_shaken

    elif k == 2:
        if len(built_plants) < 2:
            return shake(Y_current, 1, data)
        (t1, i1), (t2, i2) = random.sample(built_plants, 2)
        Y_shaken[t1, i1], Y_shaken[t2, i2] = 0, 0
        Y_shaken[t2, i1], Y_shaken[t1, i2] = 1, 1
        return Y_shaken

    elif k == 3:
        sites_with_plants = {i for _, i in built_plants}
        available_sites = [i for i in I if i not in sites_with_plants]
        can_add = len(available_sites) > 0
        can_remove = len(built_plants) > 0

        if can_add and (not can_remove or random.random() < 0.5):
            t_add = random.randrange(len(T_list))
            i_add = random.choice(available_sites)
            Y_shaken[t_add, i_add] = 1
        elif can_remove:
            t_rem, i_rem = random.choice(built_plants)
            Y_shaken[t_rem, i_rem] = 0
        return Y_shaken

    else:  # k > 3
        Y_temp = Y_shaken
        for _ in range(k - 2):
            Y_temp = shake(Y_temp, 1, data)
        return Y_temp


# ------------------------------------------
# 3) Second-stage subproblem (multi-product)
# ------------------------------------------
def create_subproblem_model(data):
    """Creates the Gurobi subproblem for multiple products."""
    sub_model = Model("SecondStage_MultiProduct")
    sub_model.setParam('OutputFlag', 0)
    sub_model.setParam("Method", 1)  # dual simplex good for repeated RHS updates

    # Keep the edited block exactly:
    sub_model.setParam("Presolve", 0)  # sometimes helps repeated RHS updates
    sub_model.setParam("Crossover", 0) # if using simplex

    S, T, P, prob, pO, pC, d, eO, eC, E_max, buy_price, sold_price = (
        data['S'], data['T'], data['P'], data['prob'], data['pO'], data['pC'], data['d'],
        data['eO'], data['eC'], data['E_max'], data['buy_price'], data['sold_price']
    )

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
            for p in P:
                sub_model.addConstr(XO[s, t, p] + XC[s, t, p] == d[s, t, p],
                                    name=f"demand_{s}_{t}_{p}")

            sub_model.addConstr(
                quicksum(eO[s, t, p] * XO[s, t, p] + eC[s, t, p] * XC[s, t, p] for p in P)
                + Sold[s, t] <= E_max[t] + Buy[s, t],
                name=f"emission_cap_{s}_{t}"
            )
            sub_model.addConstr(Sold[s, t] <= E_max[t], name=f"emission_sold_and_max_{s}_{t}")

    # Placeholder capacity constraints, RHS updated in evaluate_solution()
    capacity_constrs = {(s, t): sub_model.addConstr(quicksum(XO[s, t, p] for p in P) <= 0,
                                                    name=f"cap_{s}_{t}")
                        for s in S for t in T}

    sub_model.update()
    return sub_model, capacity_constrs


# ------------------------------------------
# 4) Evaluation with RHS capacity update
# ------------------------------------------
def evaluate_solution(Y, sub_model, capacity_constrs, data):
    """Evaluates a given Y for the multi-product problem."""
    T, I, S, b, V, u, NB_T = data['T'], data['I'], data['S'], data['b'], data['V'], data['u'], data['NB_T']
    T_list = list(T); T_list.sort()
    tpos = {t: idx for idx, t in enumerate(T_list)}

    # First-stage cost computed positionally (maintenance remaining periods)
    first_stage_cost = 0.0
    for t in T_list:
        idx = tpos[t]
        for i in I:
            if Y[idx, i] == 1:
                install_cost = V * b[i]
                remaining_periods = NB_T - idx
                maint_cost = remaining_periods * u * b[i]
                first_stage_cost += install_cost + maint_cost

    # Cumulative capacity per period index
    cumulative_capacity = {}
    running = 0.0
    for idx, t in enumerate(T_list):
        # add capacity installed at this period index
        running += sum(b[i] * Y[idx, i] for i in I)
        cumulative_capacity[t] = running

    # Update RHS
    for s in S:
        for t in T_list:
            capacity_constrs[(s, t)].RHS = float(cumulative_capacity[t])

    sub_model.optimize()
    if sub_model.status == GRB.OPTIMAL:
        return first_stage_cost + sub_model.ObjVal
    return float('inf')


# --------------------------
# 5) Q-learning helpers
# --------------------------
def initialize_q_table(states, actions):
    return {s: {a: 0.0 for a in actions} for s in states}

def choose_action(state, q_table, actions, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    q_values = q_table[state]
    max_q = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q]
    return random.choice(best_actions)

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    max_next_q = max(q_table[next_state].values())
    learned_value = reward + gamma * max_next_q
    old_value = q_table[state][action]
    q_table[state][action] = (1 - alpha) * old_value + alpha * learned_value


# --------------------------
# 6) Main Q-learning VNS
# --------------------------
def Q_VNS(max_iterations, data, k_max=K_MAX, alpha=0.1, gamma=0.95, epsilon=0.9):
    print("--- Starting Q-Learning VNS for Multi-Product Problem ---")
    start_vns_time = time.time()

    # Q-learning setup
    states = [0, 1]  # 0: Improvement, 1: Stagnation
    actions = list(range(1, k_max + 1))
    q_table = initialize_q_table(states, actions)

    # VNS initialization
    sub_model, capacity_constrs = create_subproblem_model(data)
    Y_best = generate_greedy_initial_solution(data)
    cost_best = evaluate_solution(Y_best, sub_model, capacity_constrs, data)

    hist = [cost_best]
    print(f"Initial Solution Cost: {cost_best:.2f}\n")

    current_state = 0
    for i in range(max_iterations):
        # Choose action
        action_k = choose_action(current_state, q_table, actions, epsilon)

        # Shake
        Y_shaken = shake(Y_best, action_k, data)
        cost_shaken = evaluate_solution(Y_shaken, sub_model, capacity_constrs, data)

        # Reward + next state
        if cost_shaken < cost_best:
            reward = cost_best - cost_shaken
            next_state = 0
            Y_best, cost_best = Y_shaken, cost_shaken
            print(f"Iter {i+1}: New best found (k={action_k}) -> Cost: {cost_best:.2f}")
        else:
            reward = -5.0
            next_state = 1

        hist.append(cost_best)

        # Update Q-table
        update_q_table(q_table, current_state, action_k, reward, next_state, alpha, gamma)
        current_state = next_state

        # Optional epsilon decay (usually helps)
        # epsilon = max(0.05, epsilon * 0.999)

    end_vns_time = time.time()
    print(f"\n--- Q-VNS Finished in {end_vns_time - start_vns_time:.2f} seconds ---")
    return Y_best, cost_best, hist, q_table


# --------------------------
# 7) Execution
# --------------------------
if __name__ == "__main__":
    start_time = time.process_time()

    problem_data = {
        'T': T, 'I': I, 'S': S, 'P': P,
        'NB_T': NB_T, 'NB_I': NB_I, 'NB_S': NB_S, 'NB_P': NB_P,
        'b': b, 'd': d, 'V': V, 'u': u, 'prob': prob,
        'pO': pO, 'pC': pC, 'buy_price': buy_price, 'sold_price': sold_price,
        'eO': eO, 'eC': eC, 'E_max': E_max
    }

    Y_opt, cost_opt, hist, q_table = Q_VNS(max_iterations=ITER_MAX, data=problem_data)

    with open("Computational_analysis/collection_ql.txt", "a") as file:
        file.write(f"{np.round(cost_opt, 2)}\t{np.round(time.process_time() - start_time, 2)}\n")

    plt.plot(hist)
    # plt.show()

    print("=====================================")