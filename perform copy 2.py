import numpy as np
from data import *
import time
import matplotlib.pyplot as plt
import pandas as pd
from gurobipy import Model, GRB, quicksum

start_time = time.process_time()

def plot_convergence(history):
    plt.plot(history, label="Z (Objective)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title("VNSConvergence")
    plt.legend()
    plt.grid(True)
    plt.show()



# Objective function
def Objective(Y, XO, XC, Buy, Sold):
    return(
          sum(200*b[i]*Y[t,i] for t in T for i in I) # Cout d'installation de prod. bas carbon
        + sum(20 * b[i] * Y[tp, i] for i in I for t in T for tp in range(t + 1))
        + sum(prob[s]*(pO[s,t]*XO[s,t] + pC[s,t]*XC[s,t]) for s in S for t in T) # Cout opérationnelle de production agricole
        + sum(prob[s]*(buy_price[s,t]*Buy[s,t] - sold_price[s,t]*Sold[s,t]) for s in S for t in T) # Pénalité ou gain dans la reduction ou surconsimation de carbon
    )

def Install():        
    Y = np.zeros((NB_T,NB_I))
    for i in I:
        install = np.random.randint(0,2)
        t = np.random.randint(0,NB_T)
        Y[t,i] = install
    return Y

def Allocation(Y):
    _Y = np.zeros((NB_T,NB_I))
    XO = np.zeros((NB_S,NB_T)) #qté de produits à base de LEF
    XC = np.zeros((NB_S,NB_T)) #qté de produits à base de CF
    Buy = np.zeros((NB_S,NB_T)) # nbr de crédit acheté (si on dépasse le cap)
    Sold = np.zeros((NB_S,NB_T)) #nbr de crédit vendu (si on émet moins que le cap)
    
    
    
    for i in range(NB_I):
        for t in range(NB_T):
            if Y[t, i] == 1:
                _Y[t:, i] = 1  # installation permanente à partir de t
                break  # on sort, car on a trouvé le premier t
    for s in S:
        for t in T:
            LEF_capacity = sum(b[i]*_Y[t,i] for i in I)
            Buy1, Sold1, Buy2, Sold2 = 0, 0, 0, 0 
            XO2, XO1, XC1, XC2 = 0, 0, 0, 0 
            
            XO1 = min(d[s,t], LEF_capacity)
            XO2 = 0
            XC1 = d[s,t] - XO1
            XC2 = d[s,t] - XO2
            
            emission1 = eO[s,t]*XO1 + eC[s,t]*XC1
            if emission1 > E_max[t]:
                Buy1 = emission1 - E_max[t]
            else:
                Sold1 = min(E_max[t], E_max[t] - emission1)
                
            emission2 = eO[s,t]*XO2 + eC[s,t]*XC2
            if emission2 > E_max[t]:
                Buy2 = emission2 - E_max[t]
            else:
                Sold2 = min(E_max[t], E_max[t] - emission2)
            
            XO[s,t], XC[s,t], Buy[s,t], Sold[s,t] = XO1, XC1, Buy1, Sold1
            
            Z1 = pO[s,t]*XO1 + pC[s,t]*XC1 + buy_price[s,t]*Buy1 - sold_price[s,t]*Sold1
            Z2 = pO[s,t]*XO2 + pC[s,t]*XC2 + buy_price[s,t]*Buy2 - sold_price[s,t]*Sold2
            
            if Z1 > Z2:
                XO[s,t], XC[s,t], Buy[s,t], Sold[s,t] = XO2, XC2, Buy2, Sold2

    return XO, XC, Buy, Sold


# Allocation de production
def Exact_allocation(Y):
    _Y = np.zeros((NB_T,NB_I))
    XO = np.zeros((NB_S,NB_T)) #qté de produits à base de LEF
    XC = np.zeros((NB_S,NB_T)) #qté de produits à base de CF
    Buy = np.zeros((NB_S,NB_T)) # nbr de crédit acheté (si on dépasse le cap)
    Sold = np.zeros((NB_S,NB_T)) #nbr de crédit vendu (si on émet moins que le cap)
        
    
    for i in range(NB_I):
        for t in range(NB_T):
            if Y[t, i] == 1:
                _Y[t:, i] = 1  # installation permanente à partir de t
                break  # on sort, car on a trouvé le premier t
    for s in S:
        for t in T:
            LEF_capacity = sum(b[i]*_Y[t,i] for i in I)
            model = Model("CapAndTrade_TwoStage")
            model.setParam('OutputFlag', 0)

            # === Second-stage variables for each scenario ===
            xo = model.addVar(lb=0, name="XO")  # LEF
            xc = model.addVar(lb=0, name="XC")  # CF
            buy = model.addVar(lb=0, name="Buy")
            sold = model.addVar(lb=0, name="Sold")
            
            model.setObjective(pO[s][t] * xo + pC[s][t] * xc + buy_price[s][t] * buy - sold_price[s][t] * sold, GRB.MINIMIZE)
            
            model.addConstr(xo + xc == d[s][t], name=f"demand_{s}_{t}")
            model.addConstr(xo <= LEF_capacity, name=f"capacity_{s}_{t}")
            model.addConstr(eO[s][t] * xo + eC[s][t] *xc <= E_max[t] + buy - sold, name=f"emissions_{s}_{t}")
            model.addConstr(sold <= E_max[t], name=f"emission_sold_cap_{s}_{t}")
            model.optimize()
            XO[s,t], XC[s,t], Buy[s,t], Sold[s,t] = xo.X, xc.X, buy.X, sold.X
            
    return XO, XC, Buy, Sold


def Shake(Y, k_max):
    _Y = Y.copy()
    for _ in range(k_max):
        i = np.random.randint(0, NB_I)
        if any(_Y[:,i] == 1):
            _Y[:,i] = np.zeros((NB_T))
        else:
            new_t = np.random.randint(0,NB_T)
            _Y[new_t,i] = 1
    return _Y

def VNS(ntr_max,k_max, time_limit):
    start = time.time()
    Y_best = Install()
    XO_best, XC_best, Buy_best, Sold_best = Exact_allocation(Y_best)
    Z_best = Objective(Y_best,XO_best,XC_best,Buy_best,Sold_best)
    objective_history = [Z_best]
    
    ntr = 1
    while ntr <= ntr_max and (time.time() - start) < time_limit:
        k = 1
        while k <= k_max:
            Y_local = Shake(Y_best,k)
            XO_local, XC_local, Buy_local, Sold_local = Exact_allocation(Y_local)
            
            Z_local = Objective(Y_local,XO_local,XC_local,Buy_local,Sold_local)
            
            if Z_local < Z_best:
                Y_best, XO_best, XC_best, Buy_best, Sold_best, Z_best = Y_local ,XO_local, XC_local, Buy_local, Sold_local, Z_local
                print("Amélioration tro0uvée:", Z_best)
                k = 1
                objective_history.append(Z_best)
            else:
                k += 1
        ntr += 1
    return Y_best, XO_best, XC_best, Buy_best, Sold_best, Z_best, objective_history


Y_opt, XO_opt, XC_opt, Buy_opt, Sold_opt, Z_opt, obj_history = VNS(ntr_max=500,k_max=5, time_limit=100)
# print("Installation optimal : \n",Y_opt)
emission = sum(prob[s]*(eO[s,t]*XO_opt[s,t] + eC[s,t]*XC_opt[s,t]) for s in S for t in T)
print(f"Z_vns = {np.round(Z_opt,2)}\nCO2:{np.round(emission,2)}")

# print(pd.DataFrame(Y_opt))
# print("XO_vns :")
# print(pd.DataFrame(np.round(XO_opt,2)))

# print("XC_vns")
# print(pd.DataFrame(np.round(XC_opt,2)))

# print("Buy_vns :")
# print(pd.DataFrame(np.round(Buy_opt,2)))

# print("Sold_vns")
# print(pd.DataFrame(np.round(Sold_opt,2)))
end_time = time.process_time()
print(f"Total process time : {np.round(end_time-start_time,2)} second")

plot_convergence(obj_history)