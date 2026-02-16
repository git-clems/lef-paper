from gurobipy import Model,GRB,quicksum
import sys
from Sensitivity_analysis.data2 import *
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Pour forcer les ticks entiers


r = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15])

with open('Sensitivity_analysis/plot2.py', 'a') as file:
    # file.write(f'cost = [')
    # file.write(f'ox = [')
    file.write(f'trade = [')
    
for _ in range(len(r)):
    
    u = r[_]*V
    
    model = Model("CapAndTrade_TwoStage")
    model.setParam('OutputFlag',0)

    # === First-stage decision variables ===
    Y = model.addVars(T,I,vtype=GRB.BINARY,name="Y")

    # === Second-stage variables for each scenario ===
    XO = model.addVars(S,T,P, lb=0,name="XO")  # LEF
    XC = model.addVars(S,T,P, lb=0,name="XC")  # CF
    Buy = model.addVars(S,T,lb=0,name="Buy")
    Sold = model.addVars(S,T,lb=0,name="Sold")

    # === Objective Function ===
    # First-stage cost
    install_cost = quicksum(V * b[i] * Y[t,i] for t in T for i in I)
    maintain_cost = quicksum(u * b[i] * Y[tp,i]for i in I for t in T for tp in range(t + 1))
    # Second-stage expected cost
    oper_cost = quicksum(prob[s] *(pO[s,t,j] * XO[s,t,j] + pC[s,t,j] * XC[s,t,j]) for t in T for s in S for j in P)
    carbon_trading_cost = quicksum(prob[s] *(buy_price[s,t] * Buy[s,t] - sold_price[s,t] * Sold[s,t]) for t in T for s in S)
    model.setObjective(install_cost + maintain_cost + oper_cost + carbon_trading_cost, GRB.MINIMIZE)
    

    # Demand satisfaction
    for s in S:
        for t in T:
            for j in P:
                model.addConstr(XO[s,t,j] + XC[s,t,j] == d[s,t,j],name=f"demand_{s}_{t}")
                model.addConstr(XO[s,t,j] <= quicksum(b[i] * Y[tp,i] for tp in range(t+1) for i in I),name=f"capacity_{s}_{t}")
            model.addConstr(sum(eO[s,t,j] * XO[s,t,j] + eC[s,t,j] * XC[s,t,j] for j in P) + Sold[s,t] <= E_max[t] + Buy[s,t],name=f"emissions_{s}_{t}")
            model.addConstr(Sold[s,t] <= E_max[t], name="emission_sold_and_max_{s}_{t}")
            
    # One installation per site over horizon
    for i in I:
        model.addConstr(quicksum(Y[t,i] for t in T) <= 1,name=f"install_once_{i}")

    model.optimize()
    # === Extract values after optimization ===
    XO_val = np.array([[[XO[s, t, j].X for j in P] for t in T] for s in S])
    XC_val = np.array([[[XC[s, t, j].X for j in P] for t in T] for s in S])
    Buy_val = np.array([[Buy[s, t].X for t in T] for s in S])
    Sold_val = np.array([[Sold[s, t].X for t in T] for s in S])
    Y_val = np.array([[Y[t, i].X for i in I] for t in T])

    if model.status == GRB.OPTIMAL:
        emission = sum(prob[s]*(eO[s,t,j]*XO_val[s,t,j] + eC[s,t,j]*XC_val[s,t,j]) for s in S for t in T for j in P)
        # First-stage cost
        install_cost = sum(V * b[i] * Y_val[t,i] for t in T for i in I) + sum(u * b[i] * Y_val[tp,i]for i in I for t in T for tp in range(t + 1))
        # Second-stage expected cost
        oper_cost = sum(prob[s] *(pO[s,t,j] * XO_val[s,t,j] + pC[s,t] * XC_val[s,t,j]) for t in T for s in S for j in P)
        carbon_trading_cost = sum(prob[s] *(buy_price[s,t] * Buy_val[s,t] - sold_price[s,t] * Sold_val[s,t]) for t in T for s in S)
        ox = sum(prob[s] *(XO_val[s,t,j]) for t in T for s in S for j in P)
        
        # print(emission)
        # print(model.ObjVal)
        
        with open('Sensitivity_analysis/plot2.py', 'a') as file:
            # file.write(f'{np.round(ox,2)}, ')
            file.write(f'{np.round(carbon_trading_cost,2)}, ')
            # file.write(f'{np.round(model.ObjVal,2)}, ')

    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
        model.computeIIS()
        model.write("model.ilp")
    else:
        print(f"Optimization ended with status {model.status}")
        
with open('Sensitivity_analysis/plot2.py', 'a') as file:
    file.write(f']\n')