"""
Reduced Variable Neighborhood Search (RVNS) for two-stage stochastic fertilizer planning
with a Gurobi LP subproblem reused across evaluations.

- First stage (heuristic search): installation plan Y[t,i] ∈ {0,1}
- Second stage (exact): for each scenario, period, product:
    XO/XC production + Buy/Sold carbon trading
  solved as ONE extensive-form LP across all scenarios (probability-weighted objective)

Key design choices
------------------
1) We reuse the same Gurobi LP model and ONLY update RHS of capacity constraints (fast).
2) Buy/Sold simultaneous is allowed (relaxed) but tightened with: Sold <= E_max + Buy.
3) RVNS neighborhoods preserve “install at most once per site” by construction.

Expected inputs (from Instances.instance_1 import *)
---------------------------------------------------
T, I, S, P: iterables (e.g. range)
NB_T, NB_I, NB_S, NB_P: ints
b: array-like size NB_I
d: demand array indexed [s,t,j]
V: installation cost per ton capacity
u: maintenance cost per ton capacity per period
prob: scenario probability array indexed [s]
pO, pC: unit production costs indexed [s,t,j]
buy_price, sold_price: carbon prices indexed [s,t]
eO, eC: emissions coefficients indexed [s,t,j]
E_max: cap array indexed [t]
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum

# ---------------------------
# 0) Helper: safe seed control
# ---------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

# ----------------------------------------
# 1) Greedy initial solution (multi-product)
# ----------------------------------------
def generate_greedy_initial_solution(data):
    """
    Builds an initial Y by ensuring cumulative installed capacity
    covers average total demand (summed over products) each period.
    """
    T, I, P, S = data["T"], data["I"], data["P"], data["S"]
    b, d, V, u = data["b"], data["d"], data["V"], data["u"]
    NB_T, NB_I = data["NB_T"], data["NB_I"]

    Y = np.zeros((NB_T, NB_I), dtype=int)
    site_used = [False] * NB_I

    # Average total demand per period (sum over products, average over scenarios)
    avg_total_demand = {
        t: float(sum(np.mean([d[s, t, j] for s in S]) for j in P))
        for t in T
    }

    for t in T:
        cumulative_capacity = sum(b[i] * Y[tp, i] for tp in range(t + 1) for i in I)

        while cumulative_capacity + 1e-9 < avg_total_demand[t]:
            best_i = None
            best_ratio = -1.0
            for i in I:
                if site_used[i]:
                    continue
                install_cost = V * b[i]
                # maintenance for remaining periods incl. t (matches your VNS economic assumption)
                maint_cost = (NB_T - t) * u * b[i]
                total = install_cost + maint_cost
                if total > 0:
                    ratio = b[i] / total  # benefit/cost proxy
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_i = i

            if best_i is None:
                break

            Y[t, best_i] = 1
            site_used[best_i] = True
            cumulative_capacity += b[best_i]

    return Y

# -----------------------------------
# 2) Neighborhood shake (RVNS shaking)
# -----------------------------------
def shake(Y_current, k, data):
    """
    Neighborhoods on Y only, preserving 'install at most once per site':
    k=1: move one existing installation time for one site
    k=2: swap times between two installed sites
    k=3: add one new site somewhere OR remove one installed site
    k>=4: apply k-2 times k=1
    """
    T, I = data["T"], data["I"]
    Y = Y_current.copy()

    built = list(zip(*np.where(Y == 1)))  # list of (t,i)

    if k == 1:
        if not built:
            return Y
        t_old, i = random.choice(built)
        choices = [t for t in T if t != t_old]
        if not choices:
            return Y
        t_new = random.choice(choices)
        Y[t_old, i] = 0
        Y[t_new, i] = 1
        return Y

    if k == 2:
        if len(built) < 2:
            return shake(Y_current, 1, data)
        (t1, i1), (t2, i2) = random.sample(built, 2)
        # swap install times for two different sites
        Y[t1, i1] = 0
        Y[t2, i2] = 0
        Y[t2, i1] = 1
        Y[t1, i2] = 1
        return Y

    if k == 3:
        installed_sites = {i for _, i in built}
        available_sites = [i for i in I if i not in installed_sites]
        can_add = len(available_sites) > 0
        can_remove = len(built) > 0

        if can_add and (not can_remove or random.random() < 0.5):
            t = random.choice(list(T))
            i = random.choice(available_sites)
            Y[t, i] = 1
        elif can_remove:
            t, i = random.choice(built)
            Y[t, i] = 0
        return Y

    # k >= 4: larger shake by repeated small moves
    Y_temp = Y
    for _ in range(k - 2):
        Y_temp = shake(Y_temp, 1, data)
    return Y_temp

# -----------------------------------------
# 3) Build reusable Gurobi LP subproblem once
# -----------------------------------------
def create_subproblem_model(data):
    """
    Extensive-form LP (all scenarios at once) with capacity RHS placeholders.

    Variables:
      XO[s,t,j], XC[s,t,j] >= 0
      Buy[s,t], Sold[s,t] >= 0
    Constraints:
      XO + XC = demand
      sum_j XO <= capacity(t)   [RHS updated each evaluate()]
      emissions balance with buy/sell and cap
      tightening: Sold <= E_max + Buy

    Returns: (model, capacity_constrs_dict)
    """
    sub = Model("SecondStage_LP")
    sub.setParam("OutputFlag", 0)

    # LP re-optimization friendly params
    sub.setParam("Method", 1)       # dual simplex (often best for repeated re-solves)
    sub.setParam("Presolve", 2)
    sub.setParam("Crossover", 0)

    S, T, P = data["S"], data["T"], data["P"]
    prob = data["prob"]
    pO, pC = data["pO"], data["pC"]
    buy_price, sold_price = data["buy_price"], data["sold_price"]
    d = data["d"]
    eO, eC = data["eO"], data["eC"]
    E_max = data["E_max"]

    XO = sub.addVars(S, T, P, lb=0.0, name="XO")
    XC = sub.addVars(S, T, P, lb=0.0, name="XC")
    Buy = sub.addVars(S, T, lb=0.0, name="Buy")
    Sold = sub.addVars(S, T, lb=0.0, name="Sold")

    oper = quicksum(
        prob[s] * (pO[s, t, j] * XO[s, t, j] + pC[s, t, j] * XC[s, t, j])
        for s in S for t in T for j in P
    )
    carb = quicksum(
        prob[s] * (buy_price[s, t] * Buy[s, t] - sold_price[s, t] * Sold[s, t])
        for s in S for t in T
    )
    sub.setObjective(oper + carb, GRB.MINIMIZE)

    # Demand + carbon per (s,t)
    for s in S:
        for t in T:
            for j in P:
                sub.addConstr(XO[s, t, j] + XC[s, t, j] == d[s, t, j], name=f"demand_{s}_{t}_{j}")

            # emissions balance: sum_j(eO*XO + eC*XC) <= E_max + Buy - Sold
            sub.addConstr(
                quicksum(eO[s, t, j] * XO[s, t, j] + eC[s, t, j] * XC[s, t, j] for j in P)
                <= float(E_max[t]) + Buy[s, t] - Sold[s, t],
                name=f"carbon_{s}_{t}",
            )

            # tightening (keeps relaxed buy/sell): cannot sell more than cap + purchased
            sub.addConstr(
                Sold[s, t] <= float(E_max[t]) ,
                name=f"sold_tight_{s}_{t}",
            )

    # Capacity placeholders (updated per evaluate)
    capacity_constrs = {
        (s, t): sub.addConstr(quicksum(XO[s, t, j] for j in P) <= 0.0, name=f"cap_{s}_{t}")
        for s in S for t in T
    }

    sub.update()
    return sub, capacity_constrs

# ----------------------------------------
# 4) Evaluate a Y by updating RHS and solving
# ----------------------------------------
def evaluate_solution(Y, sub_model, capacity_constrs, data):
    """
    Computes first-stage cost for Y and adds optimal second-stage LP value.
    """
    T, I, S = data["T"], data["I"], data["S"]
    b, V, u, NB_T = data["b"], data["V"], data["u"], data["NB_T"]

    # First-stage cost (installation + remaining maintenance)
    first = (
        sum(V * b[i] * Y[t, i] for t in T for i in I)
        + sum((NB_T - t) * u * b[i] * Y[t, i] for t in T for i in I)
    )

    # Update capacity RHS for each period (same across scenarios)
    cumcap = {t: float(sum(b[i] * Y[tp, i] for tp in range(t + 1) for i in I)) for t in T}
    for s in S:
        for t in T:
            capacity_constrs[(s, t)].RHS = cumcap[t]

    sub_model.optimize()

    if sub_model.status == GRB.OPTIMAL:
        return float(first + sub_model.objVal)
    return float("inf")

# -------------------
# 5) Main RVNS routine
# -------------------
def RVNS(k_max, max_iterations, data, seed=0, verbose=True):
    """
    Reduced VNS (no local search) with k=1..k_max shaking.
    Returns best Y and best cost.
    """
    set_seed(seed)
    t0 = time.time()

    sub_model, capacity_constrs = create_subproblem_model(data)

    if verbose:
        print("--- Starting RVNS ---")
        print("Generating greedy initial solution...")

    Y_best = generate_greedy_initial_solution(data)
    cost_best = evaluate_solution(Y_best, sub_model, capacity_constrs, data)

    if verbose:
        print(f"Initial cost: {cost_best:.2f}\n")

    history = [cost_best]

    for it in range(max_iterations):
        k = 1
        improved_this_iter = False

        while k <= k_max:
            Y_try = shake(Y_best, k, data)
            cost_try = evaluate_solution(Y_try, sub_model, capacity_constrs, data)

            if cost_try + 1e-9 < cost_best:
                Y_best = Y_try
                cost_best = cost_try
                improved_this_iter = True
                if verbose:
                    print(f"Iter {it:03d} | k={k} | New best: {cost_best:.2f}")
                k = 1
            else:
                k += 1

            history.append(cost_best)

        # (optional) minor diversification if stuck
        if not improved_this_iter and k_max >= 3:
            # one extra random shake at max neighborhood
            Y_try = shake(Y_best, k_max, data)
            cost_try = evaluate_solution(Y_try, sub_model, capacity_constrs, data)
            if cost_try + 1e-9 < cost_best:
                Y_best, cost_best = Y_try, cost_try
                if verbose:
                    print(f"Iter {it:03d} | diversify | New best: {cost_best:.2f}")
            history.append(cost_best)

    if verbose:
        print(f"\n--- RVNS finished in {time.time() - t0:.2f} seconds ---")

    return Y_best, cost_best, history

# ------------------------
# 6) Example main execution
# ------------------------
if __name__ == "__main__":
    # Import your instance data
    from Instances.instance_10 import *  # noqa: F401,F403

    # Build data bundle
    problem_data = {
        "T": T, "I": I, "S": S, "P": P,
        "NB_T": NB_T, "NB_I": NB_I, "NB_S": NB_S, "NB_P": NB_P,
        "b": b, "d": d, "V": V, "u": u, "prob": prob,
        "pO": pO, "pC": pC, "buy_price": buy_price, "sold_price": sold_price,
        "eO": eO, "eC": eC, "E_max": E_max,
    }

    MAX_ITER = 100
    K_MAX = 4
    start_cpu = time.process_time()

    Y_opt, cost_opt, hist = RVNS(
        k_max=K_MAX,
        max_iterations=MAX_ITER,
        data=problem_data,
        seed=0,
        verbose=True
    )

    print("\n=====================================")
    print("           Final Results")
    print("=====================================")
    print("\nBest Investment Plan (Y):")
    print(Y_opt)
    print(f"\nBest Cost Found: {cost_opt:.2f}")
    print(f"CPU time: {time.process_time() - start_cpu:.2f} sec")
    print("=====================================\n")

    plt.plot(hist)
    plt.xlabel("Evaluation step")
    plt.ylabel("Best cost so far")
    plt.title("RVNS Convergence")
    plt.show()
