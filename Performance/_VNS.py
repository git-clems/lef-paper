import numpy as np
from Data.instance_5 import *
import time
import matplotlib.pyplot as plt
import pandas as pd
from gurobipy import Model, GRB, quicksum
import random


start_time = time.process_time()

def plot_convergence(history):
    plt.plot(history, label="Z (Objective)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title("VNSConvergence")
    plt.legend()
    plt.grid(True)
    plt.show()

# === Parameters ===
V = 200
u = 70
MAX_NEIGHBORHOOD = 5
MAX_ITER = 100


problem_data = {
    'T': T, 'I': I, 'S': S, 'NB_T': NB_T, 'NB_I': NB_I, 'b': b, 'd': d, 
    'V': V, 'u': u, 'prob': prob, 'pO': pO, 'pC': pC, 
    'buy_price': buy_price, 'sold_price': sold_price, 
    'eO': eO, 'eC': eC, 'E_max': E_max
}



def generate_greedy_initial_solution(data):
    
    """
    Generates a good initial investment plan (Y matrix) using a greedy heuristic.

    Args:
        data (dict): A dictionary containing all problem data
                     (T, I, b, d, V, u, S, etc.).

    Returns:
        np.ndarray: A (len(T) x len(I)) numpy array representing the initial Y matrix.
    """
    # Unpack necessary data
    T, I, b, d, V, u, S = data['T'], data['I'], data['b'], data['d'], data['V'], data['u'], data['S']
    NB_T, NB_I = data['NB_T'], data['NB_I']

    Y_initial = np.zeros((NB_T, NB_I), dtype=int)
    sites_used = [False] * NB_I

    # Calculate average demand for each period across all scenarios
    avg_demand = {t: np.mean([d[s, t] for s in S]) for t in T}
    
    cumulative_capacity = 0.0

    # print("--- Running Greedy Heuristic ---")
    for t in T:
        # Check if we need more capacity for the current period
        while cumulative_capacity < avg_demand[t]:
            best_site_to_build = -1
            max_benefit_cost_ratio = -1

            # Find the best available site to build on *at this time t*
            for i in I:
                if not sites_used[i]:
                    # Cost = install cost + total maintenance cost from t onwards
                    install_cost = V * b[i]
                    maintenance_cost = (NB_T - t) * u * b[i]
                    total_first_stage_cost = install_cost + maintenance_cost
                    
                    if total_first_stage_cost > 0:
                        # Benefit is the capacity added
                        benefit = b[i]
                        ratio = benefit / total_first_stage_cost
                        if ratio > max_benefit_cost_ratio:
                            max_benefit_cost_ratio = ratio
                            best_site_to_build = i
            
            # If we found a site to build on
            if best_site_to_build != -1:
                i = best_site_to_build
                # print(f"Period {t}: Greedily building at site {i} (Capacity: {b[i]})")
                Y_initial[t, i] = 1
                sites_used[i] = True
                cumulative_capacity += b[i]
            else:
                # No more available sites to build
                break 
    
    # print("--- Greedy Heuristic Finished ---\n")
    return Y_initial


def shake(Y_current, k, data):
    T, I = data['T'], data['I']
    """
    Shakes a solution Y to generate a random neighbor in the k-th neighborhood.

    Args:
        Y_current (np.ndarray): The current solution matrix to be shaken.
        k (int): The neighborhood index (1, 2, 3, ...).
        data (dict): A dictionary containing problem data (T, I).

    Returns:
        np.ndarray: A new, valid Y matrix that is a neighbor of Y_current.
    """
    # NB_T, NB_I = len(T), len(I)
    Y_shaken = Y_current.copy()

    # Find locations of all existing investments as (t, i) tuples
    built_plants = list(zip(*np.where(Y_shaken == 1)))

    # --- Neighborhood N_1: Move Investment Time ---
    if k == 1:
        if not built_plants: return Y_shaken # Nothing to move

        # Pick a random plant to move
        t_old, i_to_move = random.choice(built_plants)

        # Find available new time slots for this site
        available_times = [t for t in T if t != t_old]
        if not available_times: return Y_shaken # Cannot move

        # Move it to a new random time
        t_new = random.choice(available_times)
        Y_shaken[t_old, i_to_move] = 0
        Y_shaken[t_new, i_to_move] = 1
        return Y_shaken

    # --- Neighborhood N_2: Swap Investment Times ---
    elif k == 2:
        if len(built_plants) < 2: return shake(Y_current, 1, data) # Fallback to k=1

        # Pick two different plants to swap
        (t1, i1), (t2, i2) = random.sample(built_plants, 2)
        
        # Perform the swap
        Y_shaken[t1, i1] = 0
        Y_shaken[t2, i2] = 0
        Y_shaken[t2, i1] = 1
        Y_shaken[t1, i2] = 1
        return Y_shaken

    # --- Neighborhood N_3: Add or Remove an Investment ---
    elif k == 3:
        # Find sites with no plants
        sites_with_plants = {i for _, i in built_plants}
        available_sites = [i for i in I if i not in sites_with_plants]
        
        # Decide whether to add or remove (50/50 chance if both are possible)
        can_add = len(available_sites) > 0
        can_remove = len(built_plants) > 0
        
        if can_add and (not can_remove or random.random() < 0.5): # Add
            site_to_add = random.choice(available_sites)
            time_to_add = random.choice(T)
            Y_shaken[time_to_add, site_to_add] = 1
        elif can_remove: # Remove
            t_to_remove, i_to_remove = random.choice(built_plants)
            Y_shaken[t_to_remove, i_to_remove] = 0
        
        return Y_shaken

    # --- Neighborhood N_k (k > 3): Compound Move ---
    # Perform k-2 sequential "Move Investment" shakes
    else:
        Y_temp = Y_shaken.copy()
        num_moves = k - 2 
        for _ in range(num_moves):
            # We call the k=1 logic internally to perform one move
            Y_temp = shake(Y_temp, 1, data) 
        return Y_temp


def create_subproblem_model(data):
    """Creates the Gurobi subproblem model structure. Called only ONCE."""
    sub_model = Model("SecondStageSubproblem")
    sub_model.setParam('OutputFlag', 0)
    
    # Unpack data
    S, T, prob, pO, pC, buy_price, sold_price, d, eO, eC, E_max = (
        data['S'], data['T'], data['prob'], data['pO'], data['pC'], 
        data['buy_price'], data['sold_price'], data['d'], data['eO'], data['eC'], data['E_max']
    )

    # Define all variables
    XO = sub_model.addVars(S, T, lb=0, name="XO")
    XC = sub_model.addVars(S, T, lb=0, name="XC")
    Buy = sub_model.addVars(S, T, lb=0, name="Buy")
    Sold = sub_model.addVars(S, T, lb=0, name="Sold")
    
    # Define objective
    oper_cost = quicksum(prob[s] * (pO[s,t] * XO[s,t] + pC[s,t] * XC[s,t]) for t in T for s in S)
    carbon_cost = quicksum(prob[s] * (buy_price[s,t] * Buy[s,t] - sold_price[s,t] * Sold[s,t]) for t in T for s in S)
    sub_model.setObjective(oper_cost + carbon_cost, GRB.MINIMIZE)
    
    # Add constraints that DO NOT depend on Y
    for s in S:
        for t in T:
            sub_model.addConstr(XO[s,t] + XC[s,t] == d[s,t])
            sub_model.addConstr(eO[s,t] * XO[s,t] + eC[s,t] * XC[s,t] + Sold[s,t] <= E_max[t] + Buy[s,t])

    # Add PLACEHOLDER capacity constraints. We will update the RHS later.
    capacity_constrs = {}
    for s in S:
        for t in T:
            capacity_constrs[s, t] = sub_model.addConstr(XO[s,t] <= 0, name=f"capacity_{s}_{t}")
            
    sub_model.update()
    # Return the model and the handles to the constraints we need to update
    return sub_model, capacity_constrs

def evaluate_solution(Y, sub_model, capacity_constrs, data):
    """Evaluates a given Y. Fast because it only updates and resolves."""
    # Unpack data
    T, I, b, V, u, NB_T, S = data['T'], data['I'], data['b'], data['V'], data['u'], data['NB_T'], data['S']
    
    # 1. Calculate First-Stage Cost
    install_cost = sum(V * b[i] * Y[t,i] for t in T for i in I)
    # Use the more efficient maintenance cost calculation
    maintain_cost = sum((NB_T - t) * u * b[i] * Y[t,i] for t in T for i in I)
    first_stage_cost = install_cost + maintain_cost
    
    # 2. Update and Solve Second-Stage Subproblem
    cumulative_capacity = {t: sum(b[i] * Y[tp,i] for tp in range(t+1) for i in I) for t in T}
    
    for s in S:
        for t in T:
            # THIS IS THE KEY: Update the Right-Hand Side (RHS) of the existing constraint
            capacity_constrs[s, t].RHS = cumulative_capacity[t]
    
    sub_model.optimize()
    
    if sub_model.status == GRB.OPTIMAL:
        return first_stage_cost + sub_model.ObjVal
    else:
        return float('inf') # Penalize infeasible investment plans

#Main VNS Procedure
def VNS(k_max, max_iterations,data):
    start_vns_time = time.time() # Start the master clock
    #1. Initialization
    Y_best = generate_greedy_initial_solution(problem_data) #e.g., using a greedy heuristic
    # Y_best = np.zeros((NB_T,NB_I))
    
    # 1. Create the subproblem model ONCE
    sub_model, capacity_constrs = create_subproblem_model(data)
    cost_best = evaluate_solution(Y=Y_best,data=data,sub_model=sub_model,capacity_constrs=capacity_constrs)
    # objective_history = [cost_best]
    time_history = []
    cost_history = []
    
    time_history.append(time.time() - start_vns_time)
    cost_history.append(cost_best)
    
    iter = 0
    # print("VNS started")
    while iter < max_iterations:
        k = 1
        while k <= k_max:
            # print(iter, k)
            #2. Shaking: Generate a random neighbor from the k-th neighborhood
            Y_local = shake(Y_best,k,problem_data)
            # Y_local = local_search(Y_shake, sub_model,capacity_constrs, problem_data)
            cost_local = evaluate_solution(Y=Y_local,data=data,sub_model=sub_model,capacity_constrs=capacity_constrs)
            #4. Move or not
            if cost_local < cost_best:
                Y_best = Y_local
                cost_best = cost_local
                # objective_history.append(cost_best)
                k = 1 #Success! Go back to the first neighborhood
                print(f"New best solution found: {cost_best}")
                time_history.append(time.time() - start_vns_time)
                cost_history.append(cost_best)
            else:
                k = k + 1 #No improvement, try a bigger shake
        
        iter = iter + 1
        
    return Y_best, cost_best, time_history, cost_history


Y_opt, cost_opt, t_hist_vns, c_hist_vns = VNS(k_max=4, max_iterations=500, data=problem_data)

# print(Y_opt)
print(f"Z = {np.round(cost_opt,2)}")
print(f"{NB_I}, {NB_S}, {NB_T}")
print(f"Runing time : {np.round(time.process_time() - start_time,2)}")
# print(objective_history)
