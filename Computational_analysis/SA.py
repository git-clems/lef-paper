import numpy as np
import random
import time
from gurobipy import Model, GRB, quicksum
from Instances.instance_9 import *
import matplotlib.pyplot as plt

# -----------------------------
# SA parameters
# -----------------------------
MAX_ITER = 600          # total SA iterations
T0 = 1000.0              # initial temperature
ALPHA = 0.995            # cooling rate (T <- ALPHA*T)
MIN_T = 1e-6             # stop if temperature goes below this
SEED = 0
run_time = 600


# -----------------------------
# Fixed costs (same as your code)
# -----------------------------
# V = 200
# u = 70

problem_data = {
    'T': T, 'I': I, 'S': S, 'P': P,
    'NB_T': NB_T, 'NB_I': NB_I, 'NB_S': NB_S, 'NB_P': NB_P,
    'b': b, 'd': d, 'V': V, 'u': u, 'prob': prob,
    'pO': pO, 'pC': pC, 'buy_price': buy_price, 'sold_price': sold_price,
    'eO': eO, 'eC': eC, 'E_max': E_max
}

# -----------------------------
# 1) Greedy initial solution (unchanged)
# -----------------------------
def generate_greedy_initial_solution(data):
    T, I, P, S, b, d, V, u, NB_T, NB_I = (
        data['T'], data['I'], data['P'], data['S'], data['b'], data['d'],
        data['V'], data['u'], data['NB_T'], data['NB_I']
    )

    Y_initial = np.zeros((NB_T, NB_I), dtype=int)
    sites_used = [False] * NB_I

    avg_total_demand = {
        t: np.sum([np.mean([d[s, t, j] for s in S]) for j in P])
        for t in T
    }

    for t in T:
        cumulative_capacity = sum(b[i] * Y_initial[tp, i] for tp in range(t + 1) for i in I)

        while cumulative_capacity < avg_total_demand[t]:
            best_site_to_build, max_ratio = -1, -1
            for i in I:
                if not sites_used[i]:
                    install_cost = V * b[i]
                    maintenance_cost = (NB_T - t) * u * b[i]
                    total_cost = install_cost + maintenance_cost
                    if total_cost > 0:
                        ratio = b[i] / total_cost
                        if ratio > max_ratio:
                            max_ratio = ratio
                            best_site_to_build = i

            if best_site_to_build != -1:
                i = best_site_to_build
                Y_initial[t, i] = 1
                sites_used[i] = True
                cumulative_capacity += b[i]
            else:
                break

    return Y_initial

# -----------------------------
# 2) Neighbor move (SA "move")
# -----------------------------
def neighbor_move(Y_current, data):
    """
    A single SA move (randomly chosen among several operators).
    Keeps at most one installation per site (column).
    """
    T, I = list(data['T']), list(data['I'])
    Y = Y_current.copy()

    built = list(zip(*np.where(Y == 1)))   # (t,i)
    sites_built = {i for _, i in built}
    free_sites = [i for i in I if i not in sites_built]

    op = random.choice(["move_time", "swap", "add", "remove"])

    if op == "move_time":
        if not built:
            return Y
        t_old, i = random.choice(built)
        t_new_choices = [t for t in T if t != t_old]
        if not t_new_choices:
            return Y
        t_new = random.choice(t_new_choices)
        Y[t_old, i] = 0
        Y[t_new, i] = 1
        return Y

    if op == "swap":
        if len(built) < 2:
            return Y
        (t1, i1), (t2, i2) = random.sample(built, 2)
        Y[t1, i1], Y[t2, i2] = 0, 0
        Y[t2, i1], Y[t1, i2] = 1, 1
        return Y

    if op == "add":
        if not free_sites:
            return Y
        i = random.choice(free_sites)
        t = random.choice(T)
        Y[t, i] = 1
        return Y

    # op == "remove"
    if not built:
        return Y
    t, i = random.choice(built)
    Y[t, i] = 0
    return Y

# -----------------------------
# 3) Recourse LP (same structure as your VNS)
# -----------------------------
def create_subproblem_model(data):
    sub_model = Model("SecondStageSubproblem_MultiProduct")
    sub_model.setParam('OutputFlag', 0)
    sub_model.setParam("Method", 1)
    
    # Keep the edited block exactly:
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

    oper_cost = quicksum(prob[s] * (pO[s, t, j] * XO[s, t, j] + pC[s, t, j] * XC[s, t, j])
                         for s in S for t in T for j in P)
    carbon_cost = quicksum(prob[s] * (buy_price[s, t] * Buy[s, t] - sold_price[s, t] * Sold[s, t])
                           for s in S for t in T)

    sub_model.setObjective(oper_cost + carbon_cost, GRB.MINIMIZE)

    for s in S:
        for t in T:
            for j in P:
                sub_model.addConstr(XO[s, t, j] + XC[s, t, j] == d[s, t, j])
            sub_model.addConstr(quicksum(eO[s, t, j] * XO[s, t, j] + eC[s, t, j] * XC[s, t, j] for j in P) + Sold[s, t] <= E_max[t] + Buy[s, t])
            sub_model.addConstr(Sold[s,t] <= E_max[t], name="emission_sold_and_max_{s}_{t}")
            

    capacity_constrs = {(s, t): sub_model.addConstr(quicksum(XO[s, t, j] for j in P) <= 0.0)
                        for s in S for t in T}

    sub_model.update()
    return sub_model, capacity_constrs

def evaluate_solution(Y, sub_model, capacity_constrs, data):
    T, I, b, V, u, NB_T, S = data['T'], data['I'], data['b'], data['V'], data['u'], data['NB_T'], data['S']

    first_stage_cost = sum(V * b[i] * Y[t, i] for t in T for i in I) + \
                       sum((NB_T - t) * u * b[i] * Y[t, i] for t in T for i in I)

    cumulative_capacity = {t: sum(b[i] * Y[tp, i] for tp in range(t + 1) for i in I) for t in T}
    for s in S:
        for t in T:
            capacity_constrs[(s, t)].RHS = float(cumulative_capacity[t])

    sub_model.optimize()

    if sub_model.SolCount > 0:
        return first_stage_cost + sub_model.ObjVal
    return float("inf")

# -----------------------------
# 4) Simulated Annealing
# -----------------------------
def SA(max_iter, T0, alpha, min_T, data):
    print("--- Starting Simulated Annealing (SA) ---")
    start_time = time.time()
    start = time.process_time()
    
    

    sub_model, capacity_constrs = create_subproblem_model(data)

    # Initial solution
    # Y_curr = generate_greedy_initial_solution(data)
    Y_curr = np.zeros((NB_T,NB_I))
    cost_curr = evaluate_solution(Y_curr, sub_model, capacity_constrs, data)

    Y_best = Y_curr.copy()
    cost_best = cost_curr
    hist =[cost_best]
    time_hist = [0]
    

    Ttemp = T0
    
    it = 0

    print(f"Initial cost: {cost_curr:.2f}")
    while time.process_time() - start < run_time:
        it += 1
    # for it in range(max_iter):
        if Ttemp < min_T:
            break

        # propose neighbor
        Y_new = neighbor_move(Y_curr, data)
        cost_new = evaluate_solution(Y_new, sub_model, capacity_constrs, data)

        delta = cost_new - cost_curr

        # accept if better, or with SA probability
        if delta <= 0:
            accept = True
        else:
            accept_prob = np.exp(-delta / Ttemp)
            accept = random.random() < accept_prob

        if accept:
            Y_curr = Y_new
            cost_curr = cost_new
            

            if cost_curr < cost_best:
                Y_best = Y_curr.copy()
                cost_best = cost_curr
                print(f"Iter {it}: New best -> {cost_best:.2f} | T={Ttemp:.4f}")

                hist.append(cost_best)
                time_hist.append(time.process_time() - start)
        
        # cool down
        Ttemp *= alpha

    end_time = time.time()
    print(f"--- SA Finished in {end_time - start_time:.2f} seconds ---")
    return Y_best, cost_best, hist, time_hist

# -----------------------------
# 5) Execute
# -----------------------------
if __name__ == "__main__":
    start_cpu = time.process_time()

    Y_opt, cost_opt, hist, time_hist = SA(
        max_iter=MAX_ITER,
        T0=T0,
        alpha=ALPHA,
        min_T=MIN_T,
        data=problem_data
    )

    # print("\n=====================================")
    # print("           Final Results")
    # print("=====================================")
    # print("\nBest Investment Plan (Y):")
    # print(Y_opt)

    # print(f"Total process time : {np.round(time.process_time() - start_cpu, 2)} second")
    # print(f"\nBest Cost Found: {cost_opt:.2f}")
    # print("=====================================")
    
    with open("Computational_analysis/collection_SA.txt", "a") as file:
        file.write(f"{np.round(cost_opt, 2)}\t{np.round(time.process_time() - start_cpu, 2)}\n")
    
    with open("Computational_analysis/iteration.py", "a") as interation:
        interation.write(f'hist_sa, time_sa = {hist}, {time_hist}\n')
    # plt.plot(hist)
    # plt.show()
