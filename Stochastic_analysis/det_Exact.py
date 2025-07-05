from gurobipy import Model,GRB,quicksum
from det_data import*
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Pour forcer les ticks entiers

start_time = time.process_time()

# Y = np.array([
#  [0., 0., 0., 1., 0.],
#  [0., 0., 0., 0., 0.],
#  [0., 1., 0., 0., 1.],
#  [0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0.]
# ])

d_avg = np.average(d,axis=0)
eC_avg = np.average(eC,axis=0)
eO_avg = np.average(eO,axis=0)
pC_avg = np.average(pC,axis=0)
pO_avg = np.average(pO,axis=0)
sold_price_avg = np.average(sold_price,axis=0)
buy_price_avg = np.average(buy_price,axis=0)


model = Model("CapAndTrade_TwoStage")
model.setParam('OutputFlag',0)

# === First-stage decision variables ===
Y = model.addVars(T,I,vtype=GRB.BINARY,name="Y")
# === Second-stage variables for each scenario ===
XO = model.addVars(T,lb=0,name="XO")  # LEF
XC = model.addVars(T,lb=0,name="XC")  # CF
Buy = model.addVars(T,lb=0,name="Buy")
Sold = model.addVars(T,lb=0,name="Sold")

# === Objective Function ===
# First-stage cost
install_cost = quicksum(V * b[i] * Y[t,i] for t in T for i in I)
maintain_cost = quicksum(u * b[i] * Y[tp,i]for i in I for t in T for tp in range(t + 1))
# Second-stage expected cost
oper_cost = quicksum(pO_avg[t] * XO[t] + pC_avg[t] * XC[t] for t in T)
carbon_trading_cost = quicksum(buy_price_avg[t] * Buy[t] - sold_price_avg[t] * Sold[t] for t in T)
model.setObjective(install_cost + maintain_cost + oper_cost + carbon_trading_cost, GRB.MINIMIZE)

# Demand satisfaction
for t in T:
    model.addConstr(XO[t] + XC[t] == d_avg[t], name=f"demand_{t}")
    model.addConstr(XO[t] <= quicksum(b[i] * Y[tp,i] for tp in range(t+1) for i in I), name=f"capacity_{t}")
    model.addConstr(eO_avg[t] * XO[t] + eC_avg[t] * XC[t] + Sold[t] <= E_max[t] + Buy[t], name=f"emissions_{t}")
# One installation per site over horizon
for i in I:
    model.addConstr(quicksum(Y[t,i] for t in T) <= 1, name=f"install_once_{i}")

model.optimize()
# === Extract values after optimization ===
XO_val = np.array([XO[t].X for t in T])
XC_val = np.array([XC[t].X for t in T])
Buy_val = np.array([Buy[t].X for t in T])
Sold_val = np.array([Sold[t].X for t in T])
Y_val = np.array([[Y[t,i].X for i in I] for t in T])


if model.status == GRB.OPTIMAL:
    print(Y_val)
    print(f"determinist solution : {model.ObjVal}")
    # emission = sum(eO_avg[t] * XO_val[t] + eC_avg[t] * XC_val[t] for t in T)
    # print(f"L'emission : {emission}")
    c_1 = quicksum(pO_avg[t] * XO_val[t] + pC_avg[t] * XC_val[t] for t in T)
    c_2 = quicksum(buy_price_avg[t] * Buy_val[t] - sold_price_avg[t] * Sold_val[t] for t in T)
    cost = c_1+c_2
    print("Deterministic : ",cost)
elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible.")
    model.computeIIS()
    model.write("model.ilp")
else:
    print(f"Optimization ended with status {model.status}")


end_time = time.process_time()
print(f"************* Process time ; {np.round(end_time-start_time,2)} *************" )