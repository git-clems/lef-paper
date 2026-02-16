from gurobipy import Model,GRB,quicksum
import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parents[1]))
from data import*

# from data import *

import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Pour forcer les ticks entiers

start_time = time.process_time()

# V = 200
# u = 70

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

    # print(f"Z = {np.round(model.ObjVal,2)}\nCO2 = {np.round(emission,2)}/{sum(E_max)}")
    # print(f"Installation cost : {install_cost}")
    # print(f"Production cost : {np.round(oper_cost,2)}")
    # print(f"Trading gain : {carbon_trading_cost}")
    
    # print(pd.DataFrame(Y_val))
    # print("XO :")
    # print(pd.DataFrame(np.round(XO_val,2)))

    # print("XC")
    # print(pd.DataFrame(np.round(XC_val,2)))
    
    # print("Cap")
    # print(pd.DataFrame(np.round(E_max,2)))


    # print("Emission")
    # print(pd.DataFrame(np.round(eO*XO_val + eC*XC_val,2)))

    # print("Buy_val")
    # print(pd.DataFrame(np.round(Buy_val,2)))


    # print("Sold_val")
    # print(pd.DataFrame(np.round(Sold_val,2)))
    
    plt.figure(figsize=(10, 6))
    
    index_periods = np.arange(len(T))  # Indices des périodes
    Scenario = ["S1", "S2", "S3", "S4"]
    colors = ["blue", "orange", "yellowgreen", "green"]
    bar_width = 0.2
    x = np.arange(len(T))  # positions de base des périodes (0,1,2,...)

    
    
    # index_periods = np.arange(len(T))
    # Scenario = ["Scenario 1", "FAIRE", "GOOD", "BOOM"]
    # colors = ["red", "orange", "yellowgreen", "green"]

    # plt.figure(figsize=(10, 6))

    for s_idx in S:
        x_offset = index_periods + (s_idx - (len(S)-1)/2) * 0.2  # centered offsets

        # 1D heights: total LEF production per period (sum over products)
        XO_period = np.array(
            [sum(XO_val[s_idx, t, j] for j in P) for t in T],
            dtype=float
        )

        total_lef = XO_period.sum()

        plt.bar(
            x_offset,
            XO_period,
            width=0.2,
            # label=f"{Scenario[s_idx]} : {total_lef:.2f}",
            label=f"{Scenario[s_idx]}",
            color=colors[s_idx]
        )

    for i in range(1, len(T)):
        plt.axvline(x=i - 0.5, color="black", linestyle="--", linewidth=0.5)

    plt.xlabel("Periods", fontsize=25)
    plt.ylabel("LEF based tea production (ton)", fontsize=25)
    plt.xticks(index_periods, [t + 1 for t in T], fontsize=20)
    plt.yticks(fontsize=25)
    plt.legend(title="Scenarios", fontsize=15)
    plt.tight_layout()
    plt.show()

    
        
    # === Calcul des émissions moyennes par période ===
    emission_per_period = [
        sum(prob[s] * (eO[s,t,j] * XO_val[s,t,j] + eC[s,t,j] * XC_val[s,t,j]) for s in S for j in P)
        for t in T
    ]

    # === Calcul des coûts moyens par période ===
    install_cost_per_period = [
        sum(V * b[i] * Y_val[t,i] for i in I) + sum(u * b[i] * Y_val[tp,i]for i in I for tp in range(t + 1))
        for t in T 
    ]
    product_cost_per_period = [
        sum(prob[s] *(pO[s,t,j] * XO_val[s,t,j] + pC[s,t,j] * XC_val[s,t,j]) for s in S for j in P)
        for t in T 
    ]
    trade__cost_per_period = [
        sum(prob[s] *(buy_price[s,t] * Buy_val[s,t] - sold_price[s,t] * Sold_val[s,t]) for s in S)
        for t in T 
    ]
    
    # === Figure 1 : Émissions ===
    plt.figure(figsize=(10, 5))
    plt.plot(T, emission_per_period, marker='*', color='black', label="Total emission per period")
    plt.plot(T, E_max, marker='x', color='orange', linestyle = "--", label="Emission cap per period")
    # plt.title("Émission moyenne par période")
    plt.ylabel("GHG emission (tCO₂-eq)", fontsize=25)
    plt.xlabel("Periods", fontsize=25)
    plt.xticks(x, [t+1 for t in T], fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.axhline(y=0, label = 'Carbon neutrality')
    plt.legend(fontsize=15)
    plt.show()
    
    
    bar_width = 0.6

    # Convertir en array pour simplifier les opérations
    install = np.array(install_cost_per_period)
    prod = np.array(product_cost_per_period)
    trade = np.array(trade__cost_per_period)

    # Séparer les valeurs positives et négatives du trading
    trade_pos = np.where(trade > 0, trade, 0)
    trade_neg = np.where(trade < 0, trade, 0)

    # Tracer chaque composante empilée correctement
    plt.figure(figsize=(10, 6))

    # Couche 1 : installation
    p1 = plt.bar(x, install, width=bar_width, label="LEF system installation + maintenance", color='blue')

    # Couche 2 : production empilée sur installation
    p2 = plt.bar(x, prod, width=bar_width, bottom=install, label="Tea production", color='orange')

    # Couche 3a : trading positif (empilé au-dessus du reste)
    p3a = plt.bar(x, trade_pos, width=bar_width, bottom=install + prod, label="Carbon penalty (emission > cap)", color='red')

    # Couche 3b : trading négatif (empilé vers le bas)
    p3b = plt.bar(x, trade_neg, width=bar_width, label="Carbon reward (emissiion < cap)", color='green')

    # plt.title("Coût moyen par période (barres empilées avec gains/pertes)")
    plt.ylabel("Average cost ($)", fontsize=25)
    plt.xlabel("Periods", fontsize=25)
    plt.xticks(x, [t+1 for t in T], fontsize=25)
    plt.yticks(fontsize=25)
    
    plt.grid(True, axis='y')
    plt.legend(title = "Costs" , fontsize=15)
    plt.tight_layout()
    plt.show()

elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible.")
    model.computeIIS()
    model.write("model.ilp")
else:
    print(f"Optimization ended with status {model.status}")


end_time = time.process_time()
print(f"************* Process time ; {np.round(end_time-start_time,2)} *************" )