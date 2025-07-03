import numpy as np
import time
import gymnasium as gym
from gymnasium import spaces
from data import *  # Load your model data from here
import matplotlib.pyplot as plt
import pandas as pd


# ----------------------------------------
# Core Model Functions
# ----------------------------------------

def objective_function(XO, XC, Y, Buy, Sold):
    return(
          sum(v * b[i]*Y[t,i] for t in T for i in I) # Cout d'installation de prod. bas carbon
        + sum(u * b[i] * Y[tp, i] for i in I for t in T for tp in range(t + 1))
        + sum(prob[s]*(pO[s,t]*XO[s,t] + pC[s,t]*XC[s,t]) for s in S for t in T) # Cout opérationnelle de production agricole
        + sum(prob[s]*(buy_price[s,t]*Buy[s,t] - sold_price[s,t]*Sold[s,t]) for s in S for t in T) # Pénalité ou gain dans la reduction ou surconsimation de carbon
    )
    
def Install():        
    Y = np.zeros((NB_T,NB_I))
    for i in I:
        install = np.random.randint(0,2)
        t = np.random.randint(0,NB_T)
        Y[t,i] = install
    return Y

# def Install():
#     Y = np.zeros((NB_T, NB_I), dtype=int)
#     for i in I:
#         # Choose the period with the lowest installation cost for unit i
#         t_best = min(T, key=lambda t: V[t] * b[i])
#         Y[t_best][i] = 1
#     return Y


def Allocation(Y):
    _Y = np.zeros((NB_T,NB_I))
    XO = np.zeros((NB_S,NB_T)) #qté de produits à base de LEF
    XC = np.zeros((NB_S,NB_T)) #qté de produits à base de CF
    Buy = np.zeros((NB_S,NB_T)) # nbr de crédit acheté (si on dépasse le cap)
    Sold = np.zeros((NB_S,NB_T)) #nbr de crédit vendu (si on émet moins que le cap)
    
    
    
    for i in range(NB_I):
        for t in range(NB_T):
            if Y[t, i] == 1:
                _Y[t:, i] = 1  # installation permanente à partir de t
                break  # on sort, car on a trouvé le premier t
    for s in S:
        for t in T:
            LEF_capacity = sum(b[i]*_Y[t,i] for i in I)
            Buy1, Sold1, Buy2, Sold2 = 0, 0, 0, 0 
            XO2, XO1, XC1, XC2 = 0, 0, 0, 0 
            
            XO1 = min(d[s,t], LEF_capacity)
            XO2 = 0
            XC1 = d[s,t] - XO1
            XC2 = d[s,t] - XO2
            
            emission1 = eO[s,t]*XO1 + eC[s,t]*XC1
            if emission1 > E_max[t]:
                Buy1 = emission1 - E_max[t]
            else:
                Sold1 = min(E_max[t], E_max[t] - emission1)
                
            emission2 = eO[s,t]*XO2 + eC[s,t]*XC2
            if emission2 > E_max[t]:
                Buy2 = emission2 - E_max[t]
            else:
                Sold2 = min(E_max[t], E_max[t] - emission2)
            
            XO[s,t], XC[s,t], Buy[s,t], Sold[s,t] = XO1, XC1, Buy1, Sold1
            
            Z1 = pO[s,t]*XO1 + pC[s,t]*XC1 + buy_price[s,t]*Buy1 - sold_price[s,t]*Sold1
            Z2 = pO[s,t]*XO2 + pC[s,t]*XC2 + buy_price[s,t]*Buy2 - sold_price[s,t]*Sold2
            
            if Z1 > Z2:
                XO[s,t], XC[s,t], Buy[s,t], Sold[s,t] = XO2, XC2, Buy2, Sold2

    return XO, XC, Buy, Sold


def shake(Y, k):
    Y_new = Y.copy()
    modified = set()
    for _ in range(k):
        i = np.random.randint(0, NB_I)
        if i in modified:
            continue
        t_new = np.random.randint(0, NB_T)
        Y_new[:, i] = 0
        Y_new[t_new, i] = 1
        modified.add(i)
    return Y_new

def compute_emissions(XO, XC):
    return sum(prob[s] * (eC[s, t] * XC[s][t] + eO[s, t] * XO[s][t]) for s in S for t in T)

# ----------------------------------------
# Gymnasium Environment
# ----------------------------------------

class VNSGymEnv(gym.Env):
    def __init__(self, k_max=3):
        super(VNSGymEnv, self).__init__()
        self.k_max = k_max
        self.action_space = spaces.Discrete(k_max)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        self.Y_best = None
        self.Z_best = None
        self.max_steps = 1000
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.Y_best = Install()
        XO, XC, Buy, Sold = Allocation(self.Y_best)
        self.Z_best = objective_function(XO, XC, self.Y_best, Buy, Sold)
        return np.array([self.Z_best], dtype=np.float32), {}

    def step(self, action):
        k = action + 1
        Y_new = shake(self.Y_best, k)
        XO, XC, Buy, Sold = Allocation(Y_new)
        Z_new = objective_function(XO, XC, Y_new, Buy, Sold)
        reward = self.Z_best - Z_new
        done = self.step_count >= self.max_steps

        if Z_new < self.Z_best:
            self.Y_best = Y_new
            self.Z_best = Z_new
            print(Z_new)

        self.step_count += 1
        return np.array([self.Z_best], dtype=np.float32), reward, done, False, {}

    def render(self):
        print(f"Current best cost: {self.Z_best}")

# ----------------------------------------
# Q-learning Agent
# ----------------------------------------

def train_ql_vns(env, episodes=3, alpha=0.1, gamma=0.95, epsilon=0.8):
    q_table = np.zeros((1000, env.k_max))  # 1000 cost states × k_max actions
    cost_history = []

    def discretize(cost):
        return min(int(cost // 10), 999)

    for ep in range(episodes):
        obs, _ = env.reset()
        state = discretize(obs[0])
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(env.k_max)
            else:
                action = np.argmax(q_table[state])

            obs_new, reward, done, _, _ = env.step(action)
            new_state = discretize(obs_new[0])

            q_table[state][action] += alpha * (
                reward + gamma * np.max(q_table[new_state]) - q_table[state][action]
            )
            state = new_state

            # Track best cost at every step
            cost_history.append(env.Z_best)

    return env.Y_best, env.Z_best, cost_history


# ----------------------------------------
# Run QL-VNS Optimization
# ----------------------------------------

if __name__ == "__main__":
    start_time = time.process_time()

    env = VNSGymEnv(k_max=3)
    Y_opt, Z_opt, cost_history = train_ql_vns(env, episodes=3)

    XO_opt, XC_opt, Buy_opt, Sold_opt = Allocation(Y_opt)
    emission = compute_emissions(XO_opt, XC_opt)

    print("\n===== FINAL RESULTS =====")
    print("Optimal Objective Value ===>", round(Z_opt, 3))
    print("Total Carbon Emissions ===>", round(emission, 3))
    print("Installation Plan (Y):\n", Y_opt)

    install_cost = sum(V[t] * b[i] * Y_opt[t][i] for t in T for i in I)
    print("Installation Cost: $", round(install_cost, 2))
    end_time = time.process_time()
    print("Execution Time:", round(end_time - start_time, 3), "seconds")

    # # Plot objective cost history
    # plt.figure(figsize=(10, 5))
    # plt.plot(cost_history, label="Objective value")
    # plt.title("QL-VNS Objective Value Convergence")
    # plt.xlabel("Step")
    # plt.ylabel("Objective Cost")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # Plot objective cost history with smoothing
    plt.figure(figsize=(10, 5))

    # Convert to DataFrame for rolling average
    df = pd.DataFrame({'Objective': cost_history})
    df['Smoothed'] = df['Objective'].rolling(window=50, min_periods=1).mean()

    # Plot raw and smoothed
    plt.plot(df['Objective'], color='red', alpha=0.4, label='Raw Objective Value')
    plt.plot(df['Smoothed'], color='blue', linewidth=2, label='Smoothed (Moving Avg)')

    plt.title("QL-VNS Objective Value Convergence (Smoothed)")
    plt.xlabel("Step")
    plt.ylabel("Objective Cost")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
