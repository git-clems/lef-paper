from gurobipy import Model, GRB, quicksum
from Instances.instance_5 import *
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Pour forcer les ticks entiers

model = Model("CapAndTrade_TwoStage")
model.setParam('OutputFlag',0)

# === First-stage decision variables ===
Y = model.addVars(T, I, vtype=GRB.BINARY, name="Y")
# === Second-stage variables for each scenario ===
XO = model.addVars(S, T, P, lb=0, name="XO")  # LEF
XC = model.addVars(S, T, P, lb=0, name="XC")  # CF
Buy = model.addVars(S, T, lb=0, name="Buy")
Sold = model.addVars(S, T, lb=0, name="Sold")

# === Objective Function ===
# First-stage cost
install_cost = quicksum(V * b[i] * Y[t, i] for t in T for i in I)
maintain_cost = quicksum(u * b[i] * Y[tp, i] for i in I for t in T for tp in range(t + 1))
# Second-stage expected cost
oper_cost = quicksum(prob[s] * (pO[s, t, j] * XO[s, t, j] + pC[s, t, j] * XC[s, t, j]) for t in T for s in S for j in P)
carbon_trading_cost = quicksum(prob[s] * (buy_price[s, t] * Buy[s, t] - sold_price[s, t] * Sold[s, t]) for t in T for s in S)
model.setObjective(install_cost + maintain_cost + oper_cost + carbon_trading_cost, GRB.MINIMIZE)

for s in S:
    for t in T:
        model.addConstr(quicksum(XO[s, t, j] for j in P) <= quicksum(b[i] * Y[tp, i] for tp in range(t + 1) for i in I), name=f"capacity_{s}_{t}")
        model.addConstr(quicksum(eO[s, t, j] * XO[s, t, j] + eC[s, t, j] * XC[s, t, j] for j in P) <= E_max[t] + Buy[s, t] - Sold[s, t] , name=f"emissions_{s}_{t}")
        model.addConstr(Sold[s,t] <= E_max[t], name="emission_sold_and_max_{s}_{t}")
        for j in P:
            model.addConstr(XO[s, t, j] + XC[s, t, j] == d[s, t, j], name=f"demand_{s}_{t}")
# One installation per site over horizon
for i in I:
    model.addConstr(quicksum(Y[t, i] for t in T) <= 1, name=f"install_once_{i}")


model.optimize()
# === Extract values after optimization ===
XO_val = np.array([[[XO[s, t, j].X for j in P] for t in T] for s in S])
XC_val = np.array([[[XC[s, t, j].X for j in P] for t in T] for s in S])
Buy_val = np.array([[Buy[s, t].X for t in T] for s in S])
Sold_val = np.array([[Sold[s, t].X for t in T] for s in S])
Y_val = np.array([[Y[t, i].X for i in I] for t in T])


if model.status == GRB.OPTIMAL:
    
    install = sum(V * b[i] * Y_val[t, i] for t in T for i in I) + sum(u * b[i] * Y_val[tp, i] for i in I for t in T for tp in range(t + 1))
    operation = sum(prob[s] * (pO[s, t, j] * XO_val[s, t, j] + pC[s, t, j] * XC_val[s, t, j]) for t in T for s in S for j in P)
    carbon_trading = sum(prob[s] * (buy_price[s, t] * Buy_val[s, t] - sold_price[s, t] * Sold_val[s, t]) for t in T for s in S)
    emission = sum(prob[s]*(eO[s, t, j]*XO_val[s, t, j] + eC[s, t, j]*XC_val[s, t, j]) for t in T for s in S for j in P)
    total_cost = model.ObjVal
    print(f"Total cost = {np.round(total_cost,2)}")
    print(f'LEF cost   : {np.round(install,2)}')
    print(f'Production cost     : {np.round(operation,2)}')
    print(f'carbon trading cost : {np.round(carbon_trading,2)}')
    print(f'Emission : {np.round(emission,2)}')

    with open('Stochastic_analysis/results.txt', 'a') as results:
        results.write(f"{np.round(total_cost,2)} & {np.round(install,2)} & {np.round(operation, 2)} & {np.round(carbon_trading, 2)} & {np.round(emission, 2)} \\\ \n ")
        
        
elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible.")
    model.computeIIS()
    model.write("model.ilp")
else:
    print(f"Optimization ended with status {model.status}")
