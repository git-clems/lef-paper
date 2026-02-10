from gurobipy import Model, GRB, quicksum
from Instances.instance_5 import *
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Pour forcer les ticks entiers

# k = buy_price.mean(axis=0)

# print(buy_price)
# print(k)

model = Model("CapAndTrade_TwoStage")
model.setParam('OutputFlag',0)

# === First-stage decision variables ===
Y = model.addVars(T, I, vtype=GRB.BINARY, name="Y")
# === Second-stage variables for each scenario ===
XO = model.addVars(T, P, lb=0, name="XO")  # LEF
XC = model.addVars(T, P, lb=0, name="XC")  # CF
Buy = model.addVars(T, lb=0, name="Buy")
Sold = model.addVars(T, lb=0, name="Sold")

# === Objective Function ===
# First-stage cost
install_cost = quicksum(V * b[i] * Y[t, i] for t in T for i in I)
maintain_cost = quicksum(u * b[i] * Y[tp, i] for i in I for t in T for tp in range(t + 1))
# Second-stage expected cost
oper_cost = quicksum((pO.mean(axis=0)[t, j] * XO[t, j] + pC.mean(axis=0)[t, j] * XC[t, j]) for t in T for j in P)
carbon_trading_cost = quicksum((buy_price.mean(axis=0)[t] * Buy[t] - sold_price.mean(axis=0)[t] * Sold[t]) for t in T)
model.setObjective(install_cost + maintain_cost + oper_cost + carbon_trading_cost, GRB.MINIMIZE)

for t in T:
    model.addConstr(quicksum(XO[t, j] for j in P) <= quicksum(b[i] * Y[tp, i] for tp in range(t + 1) for i in I), name=f"capacity_{t}")
    model.addConstr(quicksum(eO.mean(axis=0)[t, j] * XO[t, j] + eC.mean(axis=0)[t, j] * XC[t, j] for j in P) <= E_max[t] + Buy[t] - Sold[t] , name=f"emissions_{t}")
    model.addConstr(Sold[t] <= E_max[t], name=f"emission_sold_and_max_{t}")
    for j in P:
        model.addConstr(XO[t, j] + XC[t, j] == d.mean(axis=0)[t, j], name=f"demand_{t}_{j}")
# One installation per site over horizon
for i in I:
    model.addConstr(quicksum(Y[t, i] for t in T) <= 1, name=f"install_once_{i}")


model.optimize()
# === Extract values after optimization ===
XO_val = np.array([[XO[t, j].X for j in P] for t in T])
XC_val = np.array([[XC[t, j].X for j in P] for t in T])
Buy_val = np.array([Buy[t].X for t in T])
Sold_val = np.array([Sold[t].X for t in T])
Y_val = np.array([[Y[t, i].X for i in I] for t in T])


if model.status == GRB.OPTIMAL:
    
    install = sum(V * b[i] * Y_val[t, i] for t in T for i in I) + sum(u * b[i] * Y_val[tp, i] for i in I for t in T for tp in range(t + 1))
    operation = sum((pO.mean(axis=0)[t, j] * XO_val[t, j] + pC.mean(axis=0)[t, j] * XC_val[t, j]) for t in T for j in P)
    carbon_trading = sum((buy_price.mean(axis=0)[t] * Buy_val[t] - sold_price.mean(axis=0)[t] * Sold_val[t]) for t in T)
    emission = sum(eO.mean(axis=0)[t, j]*XO_val[t, j] + eC.mean(axis=0)[t, j]*XC_val[t, j] for t in T for j in P)
    total_cost = np.round(model.ObjVal,2)
    print(f"Total cost = {np.round(total_cost,2)}")
    print(f'LEF cost   : {np.round(install,2)}')
    print(f'Production cost     : {np.round(operation,2)}')
    print(f'carbon trading cost : {np.round(carbon_trading,2)}')
    print(f'Emission : {np.round(emission,2)}')
    
    
    with open('Stochastic_analysis/results.txt', "a") as results:
        results.write(f"{NB_I, NB_S, NB_T, NB_P} & {np.round(total_cost,2)} & {np.round(install,2)} & {np.round(operation, 2)} & {np.round(carbon_trading, 2)} & {np.round(emission, 2)} & ")
        
elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible.")
    model.computeIIS()
    model.write("model.ilp")
else:
    print(f"Optimization ended with status {model.status}")


