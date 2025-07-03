from gurobipy import Model,GRB,quicksum
from data import*
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Pour forcer les ticks entiers
from scipy.interpolate import make_interp_spline

start_time = time.process_time()
model = Model("CapAndTrade_TwoStage")
model.setParam('OutputFlag',0)

# === First-stage decision variables ===
Y = model.addVars(T,I,vtype=GRB.BINARY,name="Y")
decision = model.addVars(S,T,vtype=GRB.BINARY,name="Y")

# === Second-stage variables for each scenario ===
XO = model.addVars(S,T,lb=0,name="XO")  # LEF
XC = model.addVars(S,T,lb=0,name="XC")  # CF
Buy = model.addVars(S,T,lb=0,name="Buy")
Sold = model.addVars(S,T,lb=0,name="Sold")

v = 200
u = 70

# === Objective Function ===
# First-stage cost
install_cost = quicksum(v * b[i] * Y[t,i] for t in T for i in I)
maintain_cost = quicksum(u * b[i] * Y[tp,i]for i in I for t in T for tp in range(t + 1))
# Second-stage expected cost
oper_cost = quicksum(prob[s] *(pO[s,t] * XO[s,t] + pC[s,t] * XC[s,t]) for t in T for s in S)
carbon_trading_cost = quicksum(prob[s] *(buy_price[s,t] * Buy[s,t] - sold_price[s,t] * Sold[s,t]) for t in T for s in S)

model.setObjective(install_cost + maintain_cost + oper_cost ,GRB.MINIMIZE)



# Demand satisfaction
for s in S:
    for t in T:
        model.addConstr(XO[s,t] + XC[s,t] == d[s,t],name=f"demand_{s}_{t}")

# LEF production capacity
for s in S:
    for t in T:
        model.addConstr(XO[s,t] <= quicksum(b[i] * Y[tp,i] for tp in range(t+1) for i in I),name=f"capacity_{s}_{t}")

# One installation per site over horizon
for i in I:
    model.addConstr(quicksum(Y[t,i] for t in T) <= 1,name=f"install_once_{i}")

model.optimize()
# === Extract values after optimization ===
XO_val = np.array([[XO[s,t].X for t in T] for s in S])
XC_val = np.array([[XC[s,t].X for t in T] for s in S])
Buy_val = np.array([[Buy[s,t].X for t in T] for s in S])
Sold_val = np.array([[Sold[s,t].X for t in T] for s in S])
Y_val = np.array([[Y[t,i].X for i in I] for t in T])


if model.status == GRB.OPTIMAL:
    # emission = [sum(prob[s]*(eO[s,t]*XO_val[s,t] + eC[s,t]*XC_val[s,t]) for s in S) for t in T]
    
    # print(pd.DataFrame(Y_val))
    # print("XO :")
    # print(pd.DataFrame(np.round(XO_val,2)))
    
    # print("Emission :")
    # print(pd.DataFrame(np.round(emission,2)))

    # print("XC")
    # print(pd.DataFrame(np.round(XC_val,2)))
    
    # print("Cap")
    # print(pd.DataFrame(np.round(E_max,2)))


    # print("Emission")
    # print(pd.DataFrame(np.round(eO*XO_val + eC*XC_val,2)))

    # print("Buy")
    # print(pd.DataFrame(np.round(Buy_val,2)))
    print(model.ObjVal)
    print(Y_val)

    # print("Sell")
    # print(pd.DataFrame(np.round(Sold_val,2)))
    
elif model.status == GRB.INFEASIBLE:
    print("Model is infeasible.")
    model.computeIIS()
    model.write("model.ilp")
else:
    print(f"Optimization ended with status {model.status}")


end_time = time.process_time()
print(f"************* Process time ; {np.round(end_time-start_time,2)} *************" )