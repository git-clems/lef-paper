from gurobipy import Model, GRB, quicksum
from Instances.instance_8 import *
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Pour forcer les ticks entiers

start_time = time.process_time()

model = Model("CapAndTrade_TwoStage")
# model.setParam('OutputFlag',0)
# NB_P = 1
# P = range(NB_P)

# V = 200
# u = 70

# pO = pO.reshape(NB_S,NB_T,NB_P)
# pC = pC.reshape(NB_S,NB_T,NB_P)
# eO = eO.reshape(NB_S,NB_T,NB_P)
# eC = eC.reshape(NB_S,NB_T,NB_P)
# d = d.reshape(NB_S,NB_T,NB_P)
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
    # print(f"({NB_I}, {NB_S}, {NB_T}, {NB_P})")
    # print(f"Z = {np.round(model.ObjVal,2)}")
    # end_time = time.process_time()
    # print(f"************* Process time ; {np.round(end_time-start_time,2)} *************")
    # print(Y_val)
    
    install = sum(V * b[i] * Y_val[t, i] for t in T for i in I)
    maintain = sum(u * b[i] * Y_val[tp, i] for i in I for t in T for tp in range(t + 1))
    operation = sum(prob[s] * (pO[s, t, j] * XO_val[s, t, j] + pC[s, t, j] * XC_val[s, t, j]) for t in T for s in S for j in P)
    carbon_trading = sum(prob[s] * (buy_price[s, t] * Buy_val[s, t] - sold_price[s, t] * Sold_val[s, t]) for t in T for s in S)
    
    # print(f'Installation cost   : {np.round(install,2)}')
    # print(f'Maintenance cost    : {np.round(maintain,2)}')
    # print(f'Production cost     : {np.round(operation,2)}')
    # print(f'carbon trading cost : {np.round(carbon_trading,2)}')

    with open(f"Computational_analysis/collection_gurobi.txt", "a") as file:
        file.write(f"{np.round(time.process_time()-start_time,2)}\t")

    # index_periods = np.arange(len(T))
    # Scenario = ["POOR", "FAIRE", "GOOD", "BOOM"]
    # colors = ["red", "orange", "yellowgreen", "green"]

    # plt.figure(figsize=(10, 6))

    # for s_idx in S:
    #     x_offset = index_periods + (s_idx - (len(S)-1)/2) * 0.2  # centered offsets

    #     # 1D heights: total LEF production per period (sum over products)
    #     XO_period = np.array(
    #         [sum(XO_val[s_idx, t, j] for j in P) for t in T],
    #         dtype=float
    #     )

    #     total_lef = XO_period.sum()

    #     plt.bar(
    #         x_offset,
    #         XO_period,
    #         width=0.2,
    #         label=f"{Scenario[s_idx]} : {total_lef:.2f} ton LEF-based tea",
    #         color=colors[s_idx]
    #     )

    # for i in range(1, len(T)):
    #     plt.axvline(x=i - 0.5, color="black", linestyle="--", linewidth=0.5)

    # plt.xlabel("Periods", fontsize=25)
    # plt.ylabel("LEF based tea production (ton)", fontsize=25)
    # plt.xticks(index_periods, [t + 1 for t in T], fontsize=20)
    # plt.yticks(fontsize=25)
    # plt.legend(title="Scenario", fontsize=12)
    # plt.tight_layout()
    # plt.show()


elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible.")
    model.computeIIS()
    model.write("model.ilp")
else:
    print(f"Optimization ended with status {model.status}")
