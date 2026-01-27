"""
SA-QL (Simulated Annealing + Q-learning) using Gymnasium
=======================================================

- First-stage decision: Y[t,i] (investment schedule; at most one install per site)
- Second-stage recourse: solved exactly as an LP with Gurobi (reused model, RHS updates)
- SA: accepts worse moves with probability exp(-Δ/T)
- Q-learning: learns to choose (neighborhood k, cooling multiplier) at each step

Assumptions:
- Instances.instance_8 provides: T, I, S, P, NB_T, NB_I, NB_S, NB_P,
  b, d, prob, pO, pC, buy_price, sold_price, eO, eC, E_max
- T, I, S, P are iterable over integer indices compatible with array-like access.
"""

import numpy as np
import random
import time

from gurobipy import Model, GRB, quicksum
import gymnasium as gym
from gymnasium import spaces

from Instances.instance_6 import *

# -----------------------------
# Fixed first-stage costs
# -----------------------------
# V, u = 200, 70

problem_data = {
    "T": T, "I": I, "S": S, "P": P,
    "NB_T": NB_T, "NB_I": NB_I, "NB_S": NB_S, "NB_P": NB_P,
    "b": b, "d": d, "V": V, "u": u, "prob": prob,
    "pO": pO, "pC": pC, "buy_price": buy_price, "sold_price": sold_price,
    "eO": eO, "eC": eC, "E_max": E_max
}

# ============================================================
# 1) Greedy initial solution (multi-product)
# ============================================================
def generate_greedy_initial_solution(data):
    """Generates an initial investment plan based on average total demand across products."""
    T, I, P, S = data["T"], data["I"], data["P"], data["S"]
    b, d = data["b"], data["d"]
    V, u = data["V"], data["u"]
    NB_T, NB_I = data["NB_T"], data["NB_I"]

    Y_initial = np.zeros((NB_T, NB_I), dtype=int)
    sites_used = [False] * NB_I

    # avg total demand per period (sum over products)
    avg_total_demand = {
        t: float(np.sum([np.mean([d[s, t, j] for s in S]) for j in P]))
        for t in T
    }

    for t in T:
        cumulative_capacity = sum(
            b[i] * Y_initial[tp, i]
            for tp in range(t + 1)
            for i in I
        )

        while cumulative_capacity < avg_total_demand[t]:
            best_site, best_ratio = -1, -1
            for i in I:
                if not sites_used[i]:
                    install_cost = V * b[i]
                    maintenance_cost = (NB_T - t) * u * b[i]
                    total_cost = install_cost + maintenance_cost
                    if total_cost > 0:
                        ratio = b[i] / total_cost
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_site = i

            if best_site != -1:
                Y_initial[t, best_site] = 1
                sites_used[best_site] = True
                cumulative_capacity += b[best_site]
            else:
                break

    return Y_initial

# ============================================================
# 2) Neighborhood operator (same as your VNS shake)
# ============================================================
def shake(Y_current, k, data):
    T, I = data["T"], data["I"]
    T_list = list(T)  # safe for random.choice

    Y_shaken = Y_current.copy()
    built_plants = list(zip(*np.where(Y_shaken == 1)))  # list of (t,i)

    if k == 1:
        if not built_plants:
            return Y_shaken
        t_old, i_to_move = random.choice(built_plants)
        available_times = [t for t in T_list if t != t_old]
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
            Y_shaken[random.choice(T_list), random.choice(available_sites)] = 1
        elif can_remove:
            t_rem, i_rem = random.choice(built_plants)
            Y_shaken[t_rem, i_rem] = 0

    else:
        Y_temp = Y_shaken
        for _ in range(k - 2):
            Y_temp = shake(Y_temp, 1, data)
        return Y_temp

    return Y_shaken

# ============================================================
# 3) Recourse LP model builder (reused across evaluations)
# ============================================================
def create_subproblem_model(data):
    """Creates the Gurobi subproblem model structure for multiple products."""
    sub_model = Model("SecondStageSubproblem_MultiProduct")
    sub_model.setParam("OutputFlag", 0)
    sub_model.setParam("Method", -1)  # let Gurobi decide (good for repeated LP resolves)

    S, T, P = data["S"], data["T"], data["P"]
    prob, pO, pC = data["prob"], data["pO"], data["pC"]
    buy_price, sold_price = data["buy_price"], data["sold_price"]
    d, eO, eC, E_max = data["d"], data["eO"], data["eC"], data["E_max"]

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

    # Demand & carbon constraints
    for s in S:
        for t in T:
            for j in P:
                sub_model.addConstr(XO[s, t, j] + XC[s, t, j] == d[s, t, j], name=f"demand_{s}_{t}_{j}")
            sub_model.addConstr( quicksum(eO[s, t, j] * XO[s, t, j] + eC[s, t, j] * XC[s, t, j] for j in P) + Sold[s, t] <= E_max[t] + Buy[s, t], name=f"carbon_{s}_{t}")
            sub_model.addConstr(Sold[s,t] <= E_max[t], name="emission_sold_and_max_{s}_{t}")
            

    # Placeholder capacity constraints (RHS updated per evaluation)
    capacity_constrs = {
        (s, t): sub_model.addConstr(quicksum(XO[s, t, j] for j in P) <= 0.0, name=f"cap_{s}_{t}")
        for s in S for t in T
    }

    sub_model.update()
    return sub_model, capacity_constrs

def evaluate_solution(Y, sub_model, capacity_constrs, data):
    """Evaluate a given first-stage plan Y by updating RHS and solving recourse LP."""
    T, I, S = data["T"], data["I"], data["S"]
    b, V, u = data["b"], data["V"], data["u"]
    NB_T = data["NB_T"]

    # First-stage cost: install + maintenance from install time to end
    first_stage_cost = (
        sum(V * b[i] * Y[t, i] for t in T for i in I)
        + sum((NB_T - t) * u * b[i] * Y[t, i] for t in T for i in I)
    )

    cumulative_capacity = {
        t: sum(b[i] * Y[tp, i] for tp in range(t + 1) for i in I)
        for t in T
    }
    for s in S:
        for t in T:
            capacity_constrs[(s, t)].RHS = float(cumulative_capacity[t])

    sub_model.optimize()

    # Robust: accept any feasible LP solution
    if sub_model.SolCount > 0:
        return first_stage_cost + sub_model.ObjVal
    return float("inf")

# ============================================================
# 4) SA-QL Gymnasium environment
# ============================================================
class SAQLEnv(gym.Env):
    """
    SA-QL environment.
    Observation (state): 0 (stagnate) or 1 (improve-best)
    Action: choose (k, cooling_multiplier) encoded as single discrete action.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data,
        K_MAX=4,
        max_steps=200,
        T0=1000.0,
        Tmin=1e-6,
        cooling_choices=(0.999, 0.995, 0.99, 0.98),
        reward_success=1.0,
        reward_fail=-0.2,
        reward_scale_by_improvement=False,
        seed=0,
    ):
        super().__init__()

        self.data = data
        self.K_MAX = int(K_MAX)
        self.max_steps = int(max_steps)

        self.T0 = float(T0)
        self.Tmin = float(Tmin)
        self.cooling_choices = list(cooling_choices)

        self.reward_success = float(reward_success)
        self.reward_fail = float(reward_fail)
        self.reward_scale_by_improvement = bool(reward_scale_by_improvement)

        # action encodes: k in {1..K_MAX}, cool in cooling_choices
        self.n_cool = len(self.cooling_choices)
        self.action_space = spaces.Discrete(self.K_MAX * self.n_cool)
        self.observation_space = spaces.Discrete(2)

        random.seed(seed)
        np.random.seed(seed)

        # Reuse recourse LP model
        self.sub_model, self.capacity_constrs = create_subproblem_model(self.data)

        # internal
        self.steps = 0
        self.state = 1
        self.Ttemp = self.T0

        self.Y_curr = None
        self.cost_curr = None
        self.Y_best = None
        self.cost_best = None

    def _decode_action(self, action):
        a = int(action)
        k_index = a // self.n_cool          # 0..K_MAX-1
        cool_index = a % self.n_cool        # 0..n_cool-1
        k = k_index + 1                    # 1..K_MAX
        cool = self.cooling_choices[cool_index]
        return k, cool

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.state = 1
        self.Ttemp = self.T0

        self.Y_curr = generate_greedy_initial_solution(self.data)
        self.cost_curr = evaluate_solution(self.Y_curr, self.sub_model, self.capacity_constrs, self.data)

        self.Y_best = self.Y_curr.copy()
        self.cost_best = self.cost_curr

        return self.state, {"cost_best": self.cost_best, "T": self.Ttemp}

    def step(self, action):
        self.steps += 1

        k, cool = self._decode_action(action)

        # candidate solution from neighborhood k
        Y_new = shake(self.Y_curr, k, self.data)
        cost_new = evaluate_solution(Y_new, self.sub_model, self.capacity_constrs, self.data)

        delta = cost_new - self.cost_curr

        # SA accept rule
        if delta <= 0:
            accept = True
        else:
            accept = (random.random() < np.exp(-delta / max(self.Ttemp, 1e-12)))

        old_best = self.cost_best
        improved_best = False

        if accept:
            self.Y_curr, self.cost_curr = Y_new, cost_new

            if self.cost_curr < self.cost_best - 1e-9:
                self.Y_best = self.Y_curr.copy()
                self.cost_best = self.cost_curr
                improved_best = True

        # Reward based on best improvement
        if improved_best:
            if self.reward_scale_by_improvement:
                gain = max(0.0, old_best - self.cost_best)
                reward = self.reward_success + 0.001 * gain
            else:
                reward = self.reward_success
            self.state = 1
        else:
            reward = self.reward_fail
            self.state = 0

        # Cooling chosen by agent
        self.Ttemp *= cool

        terminated = False
        truncated = (self.steps >= self.max_steps) or (self.Ttemp < self.Tmin)

        info = {
            "k": k,
            "cool": cool,
            "T": self.Ttemp,
            "accepted": accept,
            "improved_best": improved_best,
            "cost_best": self.cost_best,
            "cost_curr": self.cost_curr,
            "cost_candidate": cost_new,
        }
        return self.state, reward, terminated, truncated, info

# ============================================================
# 5) Tabular Q-learning + greedy run
# ============================================================
def train_q_learning(
    env,
    episodes=30,
    alpha=0.2,
    gamma=0.9,
    epsilon_start=0.8,
    epsilon_end=0.05,
    epsilon_decay=0.95,
    verbose=True,
):
    """Tabular Q-learning for discrete state/action spaces."""
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA), dtype=float)

    epsilon = float(epsilon_start)

    for ep in range(episodes):
        s, info = env.reset()
        total_reward = 0.0

        while True:
            # epsilon-greedy
            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(Q[s]))

            s2, r, terminated, truncated, info = env.step(a)
            total_reward += r

            # Q update
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * np.max(Q[s2]))

            s = s2
            if terminated or truncated:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if verbose:
            print(
                f"Episode {ep+1}/{episodes} | total_reward={total_reward:.2f} "
                f"| best_cost={info['cost_best']:.2f} | eps={epsilon:.3f}"
            )

    return Q

def run_saql_policy(env, Q, verbose=True):
    """Run one greedy episode with the learned Q-table."""
    s, info = env.reset()
    while True:
        a = int(np.argmax(Q[s]))
        s, r, terminated, truncated, info = env.step(a)

        if verbose and info["improved_best"]:
            print(
                f"Step {env.steps}: best improved -> {info['cost_best']:.2f} "
                f"| k={info['k']} | cool={info['cool']} | T={info['T']:.4f}"
            )

        if terminated or truncated:
            break

    return env.Y_best, env.cost_best

# ============================================================
# 6) Main (CPU timing)
# ============================================================
if __name__ == "__main__":
    # --- User controls ---
    K_MAX = 4
    MAX_STEPS = 100          # steps per episode
    EPISODES = 1            # increase to learn more
    T0 = 1000.0
    Tmin = 1e-6
    COOLING_CHOICES = (0.999, 0.995, 0.99, 0.98)

    total_cpu_start = time.process_time()

    env = SAQLEnv(
        data=problem_data,
        K_MAX=K_MAX,
        max_steps=MAX_STEPS,
        T0=T0,
        Tmin=Tmin,
        cooling_choices=COOLING_CHOICES,
        reward_success=10.0,
        reward_fail=-0.2,
        reward_scale_by_improvement=True,
        seed=0,
    )

    # ---- training CPU time ----
    train_cpu_start = time.process_time()
    Q = train_q_learning(
        env,
        episodes=EPISODES,
        alpha=0.2,
        gamma=0.9,
        epsilon_start=0.8,
        epsilon_end=0.05,
        epsilon_decay=0.93,
        verbose=True,
    )
    train_cpu_end = time.process_time()

    print("\nLearned Q-table:")
    print(Q)
    print(f"CPU time for SA-QL training: {train_cpu_end - train_cpu_start:.2f} seconds")

    # ---- greedy run CPU time ----
    run_cpu_start = time.process_time()
    Y_best, cost_best = run_saql_policy(env, Q, verbose=True)
    run_cpu_end = time.process_time()
    print(f"CPU time for SA-QL greedy run: {run_cpu_end - run_cpu_start:.2f} seconds")

    total_cpu_end = time.process_time()
    print(f"TOTAL CPU time (SA-QL): {total_cpu_end - total_cpu_start:.2f} seconds")

    print("\n=====================================")
    print("           SA-QL Final Results")
    print("=====================================")
    print(f"Best Cost Found: {cost_best:.2f}")
    # print("Best Investment Plan (Y):")
    # print(Y_best)
