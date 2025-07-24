import numpy as np
import random
import time
from gurobipy import Model, GRB, quicksum
from Data.instance_7 import * # ASSUMPTION: This file now contains P, NB_P, and product-indexed data

start_time = time.process_time()

# --- 1. Bundle Data for Cleanliness and Portability ---
# Added P and NB_P for products

V = 200
u = 70

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
            sub_model.addConstr(
                quicksum(eO[s,t,j]*XO[s,t,j] + eC[s,t,j]*XC[s,t,j] for j in P) + Sold[s,t] <= E_max[t] + Buy[s,t]
            )

    # --- MODIFICATION: Capacity constraint sums renewable production over all products ---
    capacity_constrs = { 
        (s,t): sub_model.addConstr(quicksum(XO[s,t,j] for j in P) <= 0) 
        for s in S for t in T 
    }
            
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
            capacity_constrs[s, t].RHS = cumulative_capacity[t]
    
    sub_model.optimize()
    
    return first_stage_cost + sub_model.ObjVal if sub_model.status == GRB.OPTIMAL else float('inf')

# --- 5. Main Reduced VNS (RVNS) Procedure ---
# --- NO CHANGES NEEDED ---
# The main loop is agnostic to the problem details as long as the function interfaces are stable.
def VNS(k_max, max_iterations, data):
    """Performs a Reduced VNS search."""
    print("--- Starting Reduced VNS for Multi-Product Problem ---")
    start_vns_time = time.time()
    
    sub_model, capacity_constrs = create_subproblem_model(data)

    print("Generating initial solution...")
    Y_best = generate_greedy_initial_solution(data)
    cost_best = evaluate_solution(Y_best, sub_model, capacity_constrs, data)
    print(f"Initial Solution Cost: {cost_best:.2f}\n")

    iter_count = 0
    while iter_count < max_iterations:
        k = 1
        while k <= k_max:
            Y_shaken = shake(Y_best, k, data)
            cost_shaken = evaluate_solution(Y_shaken, sub_model, capacity_constrs, data)
            if cost_shaken < cost_best:
                Y_best = Y_shaken
                cost_best = cost_shaken
                print(f"Iter {iter_count}: New best found (from k={k}) -> Cost: {cost_best:.2f}")
                k = 1
            else:
                k += 1
        iter_count += 1
    
    end_vns_time = time.time()
    print(f"\n--- VNS Finished in {end_vns_time - start_vns_time:.2f} seconds ---")
    return Y_best, cost_best

# --- 6. Execute the Algorithm ---
if __name__ == "__main__":
    Y_opt, cost_opt = VNS(k_max=4, max_iterations=50, data=problem_data)

    # print("\n=====================================")
    # print("           Final Results")
    # print("=====================================")
    # print("\nOptimal Investment Plan (Y):")
    # print(Y_opt)
    with open(f'Performance/Note/collection.txt',"a") as file:
        file.write(f"{np.round(cost_opt,2)} \t {np.round(time.process_time()-start_time,2)}\n")
    print(f"Total process time : {np.round(time.process_time() -start_time,2)} second")
    print(f"\nOptimal Cost Found: {cost_opt:.2f}")
    print("=====================================")