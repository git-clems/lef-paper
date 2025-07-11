from gurobipy import Model,GRB,quicksum
from Sensitivity.data import*
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Pour forcer les ticks entiers

V = 200
u = 70

BETA = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]

RESULT = np.zeros((len(BETA)))
EMISSION = np.zeros((len(BETA)))
XO = np.zeros((len(BETA)))

def DRAW(beta):
    
    model = Model("CapAndTrade_TwoStage")
    model.setParam('OutputFlag',0)

    # === First-stage decision variables ===
    Y = model.addVars(T,I,vtype=GRB.BINARY,name="Y")

    # === Second-stage variables for each scenario ===
    XO = model.addVars(S,T,lb=0,name="XO")  # LEF
    XC = model.addVars(S,T,lb=0,name="XC")  # CF
    Buy = model.addVars(S,T,lb=0,name="Buy")
    Sold = model.addVars(S,T,lb=0,name="Sold")

    # === Objective Function ===
    # First-stage cost
    install_cost = quicksum(V * b[i] * Y[t,i] for t in T for i in I)
    maintain_cost = quicksum(u * b[i] * Y[tp,i]for i in I for t in T for tp in range(t + 1))
    # Second-stage expected cost
    oper_cost = quicksum(prob[s] *(pO[s,t] * XO[s,t] + pC[s,t] * XC[s,t]) for t in T for s in S)
    carbon_trading_cost = quicksum(prob[s] *(buy_price[s,t] * Buy[s,t] - 1.001 * buy_price[s,t] * Sold[s,t]) for t in T for s in S)
    model.setObjective(install_cost + maintain_cost + oper_cost + carbon_trading_cost,GRB.MINIMIZE)

    # Demand satisfaction
    for s in S:
        for t in T:
            model.addConstr(XO[s,t] + XC[s,t] == d[s,t],name=f"demand_{s}_{t}")
            model.addConstr(XO[s,t] <= quicksum(b[i] * Y[tp,i] for tp in range(t+1) for i in I),name=f"capacity_{s}_{t}")
            model.addConstr(eO[s,t] * XO[s,t] + eC[s,t] * XC[s,t] + Sold[s,t] <= E_max[t] + Buy[s,t],name=f"emissions_{s}_{t}")
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
        emission = sum(prob[s]*(eO[s,t]*XO_val[s,t] + eC[s,t]*XC_val[s,t]) for s in S for t in T)
        Z = np.round(model.ObjVal,2)
        # First-stage cost
        # # Second-stage expected cost
        # oper_cost = sum(prob[s] *(pO[s,t] * XO_val[s,t] + pC[s,t] * XC_val[s,t]) for t in T for s in S)
        # carbon_trading_cost = sum(prob[s] *(buy_price[s,t] * Buy_val[s,t] - sold_price[s,t] * Sold_val[s,t]) for t in T for s in S)

        print(f"Z = {Z}\nCO2 = {np.round(emission,2)}/{sum(E_max)}")
        # print(f"Installation cost : {install_cost}")
        # print(f"Production cost : {np.round(oper_cost,2)}")
        # print(f"Trading gain : {carbon_trading_cost}")3
        # return Z, emission, sum(prob[s]*XO_val[s,t] for s in S for t in T)
        
        return Z, emission, sum(prob[s]*XO_val[s,t] for s in S for t in T)
        
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
        model.computeIIS()
        model.write("model.ilp")
    else:
        print(f"Optimization ended with status {model.status}")

    model.dispose()

for beta in range(len(BETA)):
    RESULT[beta], EMISSION[beta], XO[beta] = DRAW(BETA[beta])

# print(pd.DataFrame(np.round(RESULT,2)))
print(RESULT)
print(EMISSION)
print(XO)
# print(pd.DataFrame(np.round(XO,2), index=CAP, columns=REG))
# print(pd.DataFrame(np.round(EMISSION,2), index=CAP, columns=REG))