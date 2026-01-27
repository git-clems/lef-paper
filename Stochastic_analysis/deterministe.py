from gurobipy import Model,GRB,quicksum
from data import*
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Pour forcer les ticks entiers

start_time = time.process_time()

V = 200
u = 70

model = Model("CapAndTrade_TwoStage")
model.setParam('OutputFlag',0)

# === First-stage decision variables ===
Y = model.addVars(T,I,vtype=GRB.BINARY,name="Y")
decision = model.addVars(T,vtype=GRB.BINARY,name="Y")

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
oper_cost = quicksum((pO.mean(axis=0)[t] * XO[t] + pC.mean(axis=0)[t] * XC[t]) for t in T)
carbon_trading_cost = quicksum((buy_price.mean(axis=0)[t] * Buy[t] - sold_price.mean(axis=0)[t] * Sold[t]) for t in T)
model.setObjective(install_cost + maintain_cost + oper_cost + carbon_trading_cost,GRB.MINIMIZE)

# Demand satisfaction
for t in T:
    model.addConstr(XO[t] + XC[t] == d.mean(axis=0)[t],name=f"demand_{t}")
    model.addConstr(XO[t] <= quicksum(b[i] * Y[tp,i] for tp in range(t+1) for i in I),name=f"capacity_{t}")
    model.addConstr(eO.mean(axis=0)[t] * XO[t] + eC.mean(axis=0)[t] * XC[t] + Sold[t] <= E_max[t] + Buy[t])
# One installation per site over horizon
for i in I:
    model.addConstr(quicksum(Y[t,i] for t in T) <= 1,name=f"install_once_{i}")

model.optimize()
# === Extract values after optimization ===
XO_val = np.array([XO[t].X for t in T])
XC_val = np.array([XC[t].X for t in T])
Buy_val = np.array([Buy[t].X for t in T])
Sold_val = np.array([Sold[t].X for t in T])
Y_val = np.array([[Y[t,i].X for i in I] for t in T])


if model.status == GRB.OPTIMAL:
    emission = sum((eO.mean(axis=0)[t]*XO_val[t] + eC.mean(axis=0)[t]*XC_val[t]) for t in T)
    # First-stage cost
    install_cost = sum(V * b[i] * Y_val[t,i] for t in T for i in I) + sum(u * b[i] * Y_val[tp,i]for i in I for t in T for tp in range(t + 1))
    # Second-stage expected cost
    oper_cost = sum((pO.mean(axis=0)[t] * XO_val[t] + pC.mean(axis=0)[t] * XC_val[t]) for t in T)
    carbon_trading_cost = sum((buy_price.mean(axis=0)[t] * Buy_val[t] - sold_price.mean(axis=0)[t] * Sold_val[t]) for t in T)

    print(f"Z = {np.round(model.ObjVal,2)}\nCO2 = {np.round(emission,2)}/{sum(E_max)}")
    print(f"Installation cost : {install_cost}")
    print(f"Production cost : {np.round(oper_cost,2)}")
    print(f"Trading gain : {carbon_trading_cost}")
    
    print(pd.DataFrame(Y_val))
    # print("XO :")
    # print(pd.DataFrame(np.round(XO_val,2)))

    # print("XC")
    # print(pd.DataFrame(np.round(XC_val,2)))
    
    print("Cap")
    print(pd.DataFrame(np.round(E_max,2)))


    print("Emission")
    print(pd.DataFrame(np.round(eO*XO_val + eC*XC_val,2)))

    print("Buy_val")
    print(pd.DataFrame(np.round(Buy_val,2)))


    print("Sold_val")
    print(pd.DataFrame(np.round(Sold_val,2)))
    
    x = np.arange(len(T))  # Indices des périodes

    plt.figure(figsize=(10, 6))

    bar_width = 0.2
    x = np.arange(len(T))  # positions de base des périodes (0,1,2,...)

    # Calcul des émissions pour chaque scénario et chaque période
    emissions = np.array([
        [eO.mean(axis=0)[t] * XO_val[t] + eC.mean(axis=0)[t] * XC_val[t] for t in T]
    ])

    # Affichage des barres côte à côte
    # x_offset = x + (s_idx - 1.5) * bar_width  # décalage horizontal
    # plt.bar(
    #     x_offset,
    #     XO_val[s_idx],
    #     width=bar_width,
    #     label=f'{Scenario[s_idx]} : {np.round(sum(XO_val[s_idx]),2)} ton of LEF based tea',
    #     color=colors[s_idx]
    # )

    # Ajouter des lignes verticales entre les périodes
    for i in range(1, len(T)):
        plt.axvline(x=i - 0.5, color='black', linestyle='--', linewidth=0.5)

    plt.xlabel("Periods",  fontsize=25)
    plt.ylabel("LEF based tea production (ton)",  fontsize=25)
    plt.xticks(x, [t + 1 for t in T], fontsize = 20)  # afficher périodes à partir de 1
    plt.yticks(fontsize=25)  # afficher périodes à partir de 1
    plt.legend(title="Scenario",  fontsize=12)
    plt.tight_layout()
    plt.show()

    
        
    # === Calcul des émissions moyennes par période ===
    emission_per_period = [eO.mean(axis=0)[t] * XO_val[t] + eC.mean(axis=0)[t] * XC_val[t]for t in T]

    # === Calcul des coûts moyens par période ===
    install_cost_per_period = [
        sum(V * b[i] * Y_val[t,i] for i in I) + sum(u * b[i] * Y_val[tp,i]for i in I for tp in range(t + 1))
        for t in T 
    ]
    product_cost_per_period = [
        pO.mean(axis=0)[t] * XO_val[t] + pC.mean(axis=0)[t] * XC_val[t]
        for t in T 
    ]
    trade__cost_per_period = [
        buy_price.mean(axis=0)[t] * Buy_val[t] - sold_price.mean(axis=0)[t] * Sold_val[t]
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
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.axhline(y=0)
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
    p3a = plt.bar(x, trade_pos, width=bar_width, bottom=install + prod, label="Cost penalty (emission > cap)", color='red')

    # Couche 3b : trading négatif (empilé vers le bas)
    p3b = plt.bar(x, trade_neg, width=bar_width, label="Cost reduction (emissiion < cap)", color='green')

    # plt.title("Coût moyen par période (barres empilées avec gains/pertes)")
    plt.ylabel("Average cost ($)", fontsize=25)
    plt.xlabel("Periods", fontsize=25)
    plt.xticks(x, [t+1 for t in T], fontsize=25)
    plt.yticks(fontsize=25)
    
    plt.grid(True, axis='y')
    plt.legend(title = "Costs")
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