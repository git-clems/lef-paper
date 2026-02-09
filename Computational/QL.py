"""
FULL QL-VNS (Q-Learning guided Variable Neighborhood Search) — Multi-product version

What this does
--------------
- First stage: binary investment plan Y[t,i] (install LEF capacity i at period t).
- Second stage (LP, solved by Gurobi): production & carbon trading per scenario/period/product,
  with a capacity RHS updated from Y.
- VNS: "shake" generates a candidate Y' in neighborhood k.
- Q-Learning: learns which neighborhood k to pick depending on state (IMPROVE vs STAGNATE).

Key features (reviewer-safe)
----------------------------
- Reuses ONE Gurobi model; only updates RHS of capacity constraints each evaluation.
- Allows relaxed buy/sell simultaneously, with a tightening bound Sold <= E_max + Buy.
- Epsilon-greedy with decay for meaningful learning.
- Reward can be constant or improvement-scaled (default: improvement-scaled).

Prereqs
-------
pip install gymnasium (optional; not required here)
gurobipy installed & licensed
Your instance file must define:
  T, I, S, P, NB_T, NB_I, NB_S, NB_P,
  b, d, V, u, prob, pO, pC, buy_price, sold_price, eO, eC, E_max

Usage
-----
from Instances.instance_5 import *
run this file.
"""

import numpy as np
import random
import time
from gurobipy import Model, GRB, quicksum
from matplotlib import pyplot as plt

MAX_ITER = 70
K_MAX = 3

# ============================================================
# 0) DATA WRAPPER
# ============================================================
def make_problem_data(**kwargs):
    return dict(kwargs)


# ============================================================
# 1) GREEDY INITIAL SOLUTION
# ============================================================
def generate_greedy_initial_solution(data):
    """Greedy initial investment plan based on average aggregate demand across products."""
    T, I, P, S = data["T"], data["I"], data["P"], data["S"]
    b, d = data["b"], data["d"]
    V, u = data["V"], data["u"]
    NB_T, NB_I = data["NB_T"], data["NB_I"]

    Y = np.zeros((NB_T, NB_I), dtype=int)
    used = [False] * NB_I

    # Average total demand per period across scenarios and products
    avg_total_demand = {
        t: float(np.sum([np.mean([d[s, t, j] for s in S]) for j in P]))
        for t in T
    }

    for t in T:
        cum_cap = sum(b[i] * Y[tp, i] for tp in range(t + 1) for i in I)
        while cum_cap < avg_total_demand[t]:
            best_i, best_ratio = -1, -1.0
            for i in I:
                if not used[i]:
                    install = V * b[i]
                    maint = (NB_T - t) * u * b[i]
                    total = install + maint
                    if total > 0:
                        ratio = b[i] / total
                        if ratio > best_ratio:
                            best_ratio, best_i = ratio, i
            if best_i == -1:
                break
            Y[t, best_i] = 1
            used[best_i] = True
            cum_cap += b[best_i]
    return Y


# ============================================================
# 2) SHAKE (NEIGHBORHOODS)
# ============================================================
def shake(Y_current, k, data):
    """Neighborhood operators on Y (install schedule). k in {1..K_MAX}."""
    T, I = data["T"], data["I"]
    Y = Y_current.copy()
    built = list(zip(*np.where(Y == 1)))  # (t,i) pairs where installed

    if k == 1:
        if not built:
            return Y
        t_old, i_move = random.choice(built)
        times = [t for t in T if t != t_old]
        if not times:
            return Y
        t_new = random.choice(times)
        Y[t_old, i_move], Y[t_new, i_move] = 0, 1
        return Y

    if k == 2:
        if len(built) < 2:
            return shake(Y_current, 1, data)
        (t1, i1), (t2, i2) = random.sample(built, 2)
        Y[t1, i1], Y[t2, i2] = 0, 0
        Y[t2, i1], Y[t1, i2] = 1, 1
        return Y

    if k == 3:
        sites_with = {i for _, i in built}
        available_sites = [i for i in I if i not in sites_with]
        can_add = len(available_sites) > 0
        can_remove = len(built) > 0
        if can_add and (not can_remove or random.random() < 0.5):
            Y[random.choice(T), random.choice(available_sites)] = 1
        elif can_remove:
            t_rem, i_rem = random.choice(built)
            Y[t_rem, i_rem] = 0
        return Y

    # k > 3: apply (k-2) times neighborhood 1
    Yt = Y
    for _ in range(k - 2):
        Yt = shake(Yt, 1, data)
    return Yt


# ============================================================
# 3) SUBPROBLEM MODEL (REUSED)
# ============================================================
def create_subproblem_model(data):
    """
    Second-stage LP (multi-product):
      Variables: XO, XC (S,T,P), Buy,Sold (S,T)
      Constraints:
        - Demand per (s,t,j): XO + XC = d
        - Emissions: sum_j eO*XO + eC*XC + Sold <= E_max + Buy
        - Tightening: Sold <= E_max + Buy
        - Capacity placeholder: sum_j XO <= RHS(s,t) (updated from Y)
    """
    sub = Model("SecondStage_MultiProduct")
    sub.setParam("OutputFlag", 0)
    # sub.setParam("Method", 1)  # dual simplex good for repeated RHS updates

    S, T, P = data["S"], data["T"], data["P"]
    prob = data["prob"]
    pO, pC = data["pO"], data["pC"]
    d = data["d"]
    eO, eC = data["eO"], data["eC"]
    E_max = data["E_max"]
    buy_price, sold_price = data["buy_price"], data["sold_price"]

    XO = sub.addVars(S, T, P, lb=0.0, name="XO")
    XC = sub.addVars(S, T, P, lb=0.0, name="XC")
    Buy = sub.addVars(S, T, lb=0.0, name="Buy")
    Sold = sub.addVars(S, T, lb=0.0, name="Sold")

    oper_cost = quicksum(
        prob[s] * (pO[s, t, j] * XO[s, t, j] + pC[s, t, j] * XC[s, t, j])
        for s in S for t in T for j in P
    )
    carbon_cost = quicksum(
        prob[s] * (buy_price[s, t] * Buy[s, t] - sold_price[s, t] * Sold[s, t])
        for s in S for t in T
    )
    sub.setObjective(oper_cost + carbon_cost, GRB.MINIMIZE)

    for s in S:
        for t in T:
            # demand per product
            for j in P:
                sub.addConstr(XO[s, t, j] + XC[s, t, j] == d[s, t, j], name=f"demand_{s}_{t}_{j}")

            # emissions
            sub.addConstr(
                quicksum(eO[s, t, j] * XO[s, t, j] + eC[s, t, j] * XC[s, t, j] for j in P)
                + Sold[s, t]
                <= E_max[t] + Buy[s, t],
                name=f"emissions_{s}_{t}"
            )

            # tightening without exclusivity binaries
            sub.addConstr(Sold[s, t] <= E_max[t] , name=f"sold_bound_{s}_{t}")

    # capacity placeholders updated during evaluation
    cap_constr = {
        (s, t): sub.addConstr(quicksum(XO[s, t, j] for j in P) <= 0.0, name=f"cap_{s}_{t}")
        for s in S for t in T
    }

    sub.update()
    return sub, cap_constr


def evaluate_solution(Y, sub_model, cap_constr, data):
    """Total cost = first stage(Y) + second-stage optimum (LP)."""
    T, I, S = data["T"], data["I"], data["S"]
    b, V, u = data["b"], data["V"], data["u"]
    NB_T = data["NB_T"]

    # first stage
    first_stage = (
        sum(V * b[i] * Y[t, i] for t in T for i in I)
        + sum((NB_T - t) * u * b[i] * Y[t, i] for t in T for i in I)
    )

    # cumulative capacity per period
    cum_cap = {t: sum(b[i] * Y[tp, i] for tp in range(t + 1) for i in I) for t in T}

    # update RHS
    for s in S:
        for t in T:
            cap_constr[(s, t)].RHS = float(cum_cap[t])

    sub_model.optimize()
    if sub_model.status != GRB.OPTIMAL:
        return float("inf")
    return float(first_stage + sub_model.ObjVal)


# ============================================================
# 4) TABULAR Q-LEARNING
# ============================================================
def initialize_q_table(states, actions):
    return {s: {a: 0.0 for a in actions} for s in states}


def choose_action(state, q_table, actions, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    # exploit
    q_vals = q_table[state]
    max_q = max(q_vals.values())
    best = [a for a, q in q_vals.items() if q == max_q]
    return random.choice(best)


def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    max_next_q = max(q_table[next_state].values())
    td_target = reward + gamma * max_next_q
    q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * td_target


# ============================================================
# 5) FULL QL-VNS ALGORITHM
# ============================================================
def QL_VNS(
    data,
    gamma: float,
    iter_max: int,
    k_max: int,
    alpha: float = 0.05,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    reward_mode: str = "scaled",  # "scaled" or "constant"
    r_success: float = 10.0,
    r_fail: float = -10.0,
    seed: int = 0,
    verbose: bool = True,
):
    """
    Paper-faithful QL-VNS:
      States: 0=IMPROVE, 1=STAGNATE
      Actions: k in {1..k_max}
      Each iteration:
        - choose k via epsilon-greedy using Q
        - shake in Nk
        - evaluate
        - reward and update Q
        - accept if improved (best-so-far strategy)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Q-learning setup
    STATES = [0, 1]
    ACTIONS = list(range(1, k_max + 1))
    Q = initialize_q_table(STATES, ACTIONS)

    # Build subproblem once
    sub_model, cap_constr = create_subproblem_model(data)

    # Initial solution
    Y_best = generate_greedy_initial_solution(data)
    cost_best = evaluate_solution(Y_best, sub_model, cap_constr, data)

    # Traces
    best_cost_trace = [cost_best]
    chosen_k_trace = []
    improve_trace = []

    # start in stagnation (more consistent)
    state = 1

    epsilon = float(epsilon_start)
    t0 = time.time()

    if verbose:
        print("--- Starting QL-VNS ---")
        print(f"Initial cost: {cost_best:.4f}")

    for it in range(iter_max):
        # 1) choose neighborhood
        k = choose_action(state, Q, ACTIONS, epsilon)

        # 2) shake + evaluate
        Y_shake = shake(Y_best, k, data)
        cost_shake = evaluate_solution(Y_shake, sub_model, cap_constr, data)

        # 3) reward + next state
        improved = cost_shake < cost_best - 1e-9
        if improved:
            if reward_mode == "scaled":
                rel_impr = (cost_best - cost_shake) / (abs(cost_best) + 1e-9)
                reward = 1.0 + 100.0 * rel_impr
            else:
                reward = cost_best - cost_shake
                # reward = r_success
            next_state = 0
            Y_best, cost_best = Y_shake, cost_shake
            if verbose:
                print(f"Iter {it+1:04d}: improve with k={k} -> best={cost_best:.4f}")
        else:
            reward = r_fail if reward_mode == "constant" else -1.0
            # reward = r_fail if reward_mode == "constant" else -0.05
            next_state = 1

        # 4) Q update
        update_q_table(Q, state, k, reward, next_state, alpha, gamma)
        state = next_state

        # 5) decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # logging
        best_cost_trace.append(cost_best)
        chosen_k_trace.append(k)
        improve_trace.append(improved)

    elapsed = time.time() - t0
    if verbose:
        print(f"--- QL-VNS finished in {elapsed:.2f}s ---")

    return {
        "Y_best": Y_best,
        "cost_best": cost_best,
        "Q_table": Q,
        "best_cost_trace": best_cost_trace,
        "k_trace": chosen_k_trace,
        "improve_trace": improve_trace,
        "elapsed_sec": elapsed,
    }


# ============================================================
# 6) RUN
# ============================================================
if __name__ == "__main__":
    from Instances.instance_7 import *  # noqa: F401,F403
    
    start_time = time.process_time()

    data = make_problem_data(
        T=T, I=I, S=S, P=P,
        NB_T=NB_T, NB_I=NB_I, NB_S=NB_S, NB_P=NB_P,
        b=b, d=d, V=V, u=u, prob=prob,
        pO=pO, pC=pC,
        buy_price=buy_price, sold_price=sold_price,
        eO=eO, eC=eC, E_max=E_max
    )

    out = QL_VNS(
        data=data,
        iter_max=500,
        k_max=3,
        alpha=0.25,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        reward_mode="scaled",   # or "constant"
        r_success=10.0,
        r_fail=-10.0,
        seed=42,
        verbose=True
    )

    # print("\n=====================================")
    # print("           QL-VNS Results")
    # print("=====================================")
    # print("Best cost:", float(out["cost_best"]))
    # print("Best Y:\n", out["Y_best"])
    # print("Elapsed (s):", round(out["elapsed_sec"], 2))
    
    
    with open(f"Computational_analysis/collection_ql.txt", "a") as file:
        file.write(f"{np.round(out["cost_best"],2)} \t {np.round(time.process_time()-start_time,2)}\n")

    # Plot progress
    plt.plot(out["best_cost_trace"])
    plt.title("QL-VNS best cost over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best cost")
    plt.show()
