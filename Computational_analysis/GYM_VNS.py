import numpy as np
import random
import time
from gurobipy import Model, GRB, quicksum
import gymnasium as gym
from gymnasium import spaces
from Instances.instance_6 import *

# V, u = 200, 70
_values = []

problem_data = {
    'T': T, 'I': I, 'S': S, 'P': P,
    'NB_T': NB_T, 'NB_I': NB_I, 'NB_S': NB_S, 'NB_P': NB_P,
    'b': b, 'd': d, 'V': V, 'u': u, 'prob': prob, 
    'pO': pO, 'pC': pC, 'buy_price': buy_price, 'sold_price': sold_price, 
    'eO': eO, 'eC': eC, 'E_max': E_max
}


# --- 2. Heuristic for Initial Solution (Modified for Multi-Product) ---
def generate_greedy_initial_solution(data):
    """Generates an initial investment plan based on total aggregate demand."""
    # Unpack data
    T, I, P, S, b, d, V, u, NB_T, NB_I = (
        data['T'], data['I'], data['P'], data['S'], data['b'], data['d'], 
        data['V'], data['u'], data['NB_T'], data['NB_I']
    )
    
    Y_initial = np.zeros((NB_T, NB_I), dtype=int)
    sites_used = [False] * NB_I
    
    # --- MODIFICATION: Calculate average TOTAL demand across all products for each period ---
    avg_total_demand = {
        t: np.sum([np.mean([d[s,t,j] for s in S]) for j in P]) 
        for t in T
    }
    
    for t in T:
        cumulative_capacity = sum(b[i] * Y_initial[tp, i] for tp in range(t + 1) for i in I)
        
        while cumulative_capacity < avg_total_demand[t]: # Check against total demand
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

# --- 3. Shake Function to Explore Neighborhoods ---
# --- NO CHANGES NEEDED ---
# This function only manipulates the Y matrix, which is not product-specific.
def shake(Y_current, k, data):
    T, I = data['T'], data['I']
    Y_shaken = Y_current.copy()
    built_plants = list(zip(*np.where(Y_shaken == 1)))
    # ... (rest of the function is identical to the single-product version) ...
    if k == 1:
        if not built_plants: return Y_shaken
        t_old, i_to_move = random.choice(built_plants)
        available_times = [t for t in T if t != t_old]
        if not available_times: return Y_shaken
        t_new = random.choice(available_times)
        Y_shaken[t_old, i_to_move], Y_shaken[t_new, i_to_move] = 0, 1
    elif k == 2:
        if len(built_plants) < 2: return shake(Y_current, 1, data)
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

# --- 4. Efficient Evaluation Functions (Modified for Multi-Product) ---
def create_subproblem_model(data):
    """Creates the Gurobi subproblem model structure for multiple products."""
    sub_model = Model("SecondStageSubproblem_MultiProduct")
    sub_model.setParam('OutputFlag', 0)
    S, T, P, prob, pO, pC, buy_price, sold_price, d, eO, eC, E_max = (
        data['S'], data['T'], data['P'], data['prob'], data['pO'], data['pC'], 
        data['buy_price'], data['sold_price'], data['d'], data['eO'], data['eC'], data['E_max']
    )
    
    # --- MODIFICATION: XO and XC are now indexed by Product P ---
    XO = sub_model.addVars(S, T, P, lb=0, name="XO")
    XC = sub_model.addVars(S, T, P, lb=0, name="XC")
    Buy, Sold = sub_model.addVars(S, T, lb=0, name="Buy"), sub_model.addVars(S, T, lb=0, name="Sold")
    
    # --- MODIFICATION: Objective now sums over products ---
    oper_cost = quicksum(prob[s] * (pO[s,t,j]*XO[s,t,j] + pC[s,t,j]*XC[s,t,j]) for s in S for t in T for j in P)
    carbon_cost = quicksum(prob[s] * (buy_price[s,t]*Buy[s,t] - sold_price[s,t]*Sold[s,t]) for s in S for t in T)
    sub_model.setObjective(oper_cost + carbon_cost, GRB.MINIMIZE)
    
    for s in S:
        for t in T:
            # --- MODIFICATION: Demand satisfaction is now per product ---
            for j in P:
                sub_model.addConstr(XO[s,t,j] + XC[s,t,j] == d[s,t,j])
            
            # --- MODIFICATION: Emissions are summed over all products ---
            sub_model.addConstr(quicksum(eO[s,t,j]*XO[s,t,j] + eC[s,t,j]*XC[s,t,j] for j in P) + Sold[s,t] <= E_max[t] + Buy[s,t])
            sub_model.addConstr(Sold[s,t] <= E_max[t], name="emission_sold_and_max_{s}_{t}")
            

    # --- MODIFICATION: Capacity constraint sums renewable production over all products ---
    capacity_constrs = { (s,t): sub_model.addConstr(quicksum(XO[s,t,j] for j in P) <= 0) for s in S for t in T }
            
    sub_model.update()
    return sub_model, capacity_constrs

def evaluate_solution(Y, sub_model, capacity_constrs, data):
    """Evaluates a given Y for the multi-product problem."""
    T, I, b, V, u, NB_T, S = data['T'], data['I'], data['b'], data['V'], data['u'], data['NB_T'], data['S']
    
    # First-stage cost calculation is UNCHANGED
    first_stage_cost = sum(V*b[i]*Y[t,i] for t in T for i in I) + \
                       sum((NB_T - t)*u*b[i]*Y[t,i] for t in T for i in I)
    
    cumulative_capacity = {t: sum(b[i] * Y[tp,i] for tp in range(t+1) for i in I) for t in T}
    for s in S:
        for t in T:
            # Update RHS of the placeholder capacity constraint. This logic is UNCHANGED.
            # capacity_constrs[s, t].RHS = cumulative_capacity[t]
            capacity_constrs[(s, t)].RHS = float(cumulative_capacity[t])

    
    sub_model.optimize()
    
    return first_stage_cost + sub_model.ObjVal if sub_model.status == GRB.OPTIMAL else float('inf')


class QLVNSEnv(gym.Env):
    """
    Gymnasium Env for QL-based VNS.
    Observation: 0 (stagnate) or 1 (improve)
    Action: choose neighborhood index k in {1..K_MAX} (encoded as 0..K_MAX-1)
    """

    metadata = {"render_modes": []}

    def __init__(self, data, K_MAX, max_steps,
                 reward_success=1.0, reward_fail=-0.2,
                 reward_scale_by_improvement=False,
                 use_current_solution=False, seed=0):
        super().__init__()

        self.data = data
        self.K_MAX = K_MAX
        self.max_steps = max_steps

        self.reward_success = reward_success
        self.reward_fail = reward_fail
        self.reward_scale_by_improvement = reward_scale_by_improvement

        # If True: shake around current solution; else shake around best (your current VNS style)
        self.use_current_solution = use_current_solution

        # spaces
        self.action_space = spaces.Discrete(K_MAX)     # actions: 0..K_MAX-1
        self.observation_space = spaces.Discrete(2)    # states: 0/1

        # RNG
        self._rng = np.random.default_rng(seed)
        random.seed(seed)

        # Gurobi subproblem model (reused each step)
        self.sub_model, self.capacity_constrs = create_subproblem_model(self.data)

        # internal
        self.steps = 0
        self.state = 1  # start optimistic
        self.Y_best = None
        self.cost_best = None
        self.Y_curr = None
        self.cost_curr = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            random.seed(seed)

        self.steps = 0
        self.state = 1  # simprove initially

        self.Y_best = generate_greedy_initial_solution(self.data)
        self.cost_best = evaluate_solution(self.Y_best, self.sub_model, self.capacity_constrs, self.data)

        # optional current solution
        self.Y_curr = self.Y_best.copy()
        self.cost_curr = self.cost_best

        obs = self.state
        info = {"cost_best": self.cost_best}
        return obs, info

    def step(self, action):
        self.steps += 1

        # map action -> neighborhood k in {1..K_MAX}
        k = int(action) + 1

        # choose base solution for shaking
        baseY = self.Y_curr if self.use_current_solution else self.Y_best
        baseCost = self.cost_curr if self.use_current_solution else self.cost_best

        # shake and evaluate
        Y_candidate = shake(baseY, k, self.data)
        cost_candidate = evaluate_solution(Y_candidate, self.sub_model, self.capacity_constrs, self.data)

        improved = cost_candidate < baseCost - 1e-9

        if improved:
            # accept into current (if using current)
            if self.use_current_solution:
                self.Y_curr = Y_candidate
                self.cost_curr = cost_candidate

            # update global best if needed
            if cost_candidate < self.cost_best - 1e-9:
                self.Y_best = Y_candidate
                self.cost_best = cost_candidate

            # reward
            if self.reward_scale_by_improvement:
                # scaled reward: larger improvement => bigger reward
                delta = max(0.0, baseCost - cost_candidate)
                reward = self.reward_success + 0.001 * delta
            else:
                reward = self.reward_success

            self.state = 1  # improve

        else:
            # no improvement
            reward = self.reward_fail
            self.state = 0  # stagnate

        terminated = False
        truncated = (self.steps >= self.max_steps)

        obs = self.state
        info = {
            "k": k,
            "cost_best": self.cost_best,
            "cost_candidate": cost_candidate,
            "improved": improved
        }
        return obs, reward, terminated, truncated, info

def train_q_learning(env, episodes=30, alpha=0.2, gamma=0.9,
                     epsilon_start=0.8, epsilon_end=0.05, epsilon_decay=0.95,
                     verbose=True):
    """
    Tabular Q-learning for small discrete state/action spaces.
    Returns learned Q-table.
    """
    nS = env.observation_space.n   # 2
    nA = env.action_space.n        # K_MAX
    Q = np.zeros((nS, nA), dtype=float)

    epsilon = epsilon_start

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
            best_next = np.max(Q[s2])
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * best_next)

            s = s2

            if terminated or truncated:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if verbose:
            print(f"Episode {ep+1}/{episodes} | total_reward={total_reward:.2f} "
                  f"| best_cost={info.get('cost_best', None):.2f} | epsilon={epsilon:.3f}")

    return Q

def run_policy(env, Q, verbose=True):
    s, info = env.reset()
    best_cost = info["cost_best"]

    while True:
        a = int(np.argmax(Q[s]))
        s, r, terminated, truncated, info = env.step(a)

        if verbose and info["improved"]:
            print(f"Step {env.steps}: improved with k={info['k']} -> best_cost={info['cost_best']:.2f}")

        best_cost = info["cost_best"]

        if terminated or truncated:
            break

    return env.Y_best, env.cost_best

if __name__ == "__main__":
    MAX_ITER = 100
    K_MAX = 4

    # Create Gym environment
    env = QLVNSEnv(
        data=problem_data,
        K_MAX=K_MAX,
        max_steps=MAX_ITER,
        reward_success=10.0,
        reward_fail=-0.2,
        reward_scale_by_improvement=True,   # optional
        use_current_solution=True,          # recommended: better exploration
        seed=0
    )
    
    cpu_start = time.process_time()
    
    # Train Q-learning
    Q = train_q_learning(
        env,
        episodes=1,
        alpha=0.2,
        gamma=0.9,
        epsilon_start=0.8,
        epsilon_end=0.05,
        epsilon_decay=0.93,
        verbose=True
    )

    print("\nLearned Q-table:")
    print(Q)

    # Run learned policy once (greedy)
    Y_best, cost_best = run_policy(env, Q, verbose=True)
    cpu_end = time.process_time()
    print(f"CPU time for Q-learning training: {cpu_end - cpu_start:.2f} seconds")

    print("\n=====================================")
    print("           QL-VNS Final Results")
    print("=====================================")
    print("Best Investment Plan (Y):")
    # print(Y_best)
    print(f"Best Cost Found: {cost_best:.2f}")
