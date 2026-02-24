import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

gurobi_data = pd.read_excel('Result_computational/full_test_data.xlsx', sheet_name='gurobi', index_col='Instance')
vns_data = pd.read_excel('Result_computational/full_test_data.xlsx', sheet_name='vns', index_col='Instance')
vns_ql_data = pd.read_excel('Result_computational/full_test_data.xlsx', sheet_name='vns-ql', index_col='Instance')
sa_data = pd.read_excel('Result_computational/full_test_data.xlsx', sheet_name='as', index_col='Instance')


gurobi_time = np.array(gurobi_data.iloc[:, 1:10]).mean(axis=1)
gurobi_cost = np.array(gurobi_data['cost'])
gurobi_gap = np.array(gurobi_data['gap'])

vns_time = np.array(vns_data.iloc[:, 1:10].mean(axis=1))
vns_cost = np.array(vns_data.iloc[:,10:].mean(axis=1))

vns_ql_time = np.array(vns_ql_data.iloc[:, 1:10].mean(axis=1))
vns_ql_cost = np.array(vns_ql_data.iloc[:,10:].mean(axis=1))

sa_time = np.array(sa_data.iloc[:, 1:10].mean(axis=1))
sa_cost = np.array(sa_data.iloc[:,10:].mean(axis=1))


with open("Result_computational/avg_result.csv","w") as file:
    file.write(f'Instance \t Gurobi \t VNS \t AS \t VNS-QL \t Gurobi \t VNS \t AS \t VNS-QL \n')
    for _ in range(gurobi_data.shape[0]):
        file.write(f'{gurobi_data.index[_]} \t& {np.round(gurobi_cost[_],2)} \t& {np.round(vns_cost[_],2)} \t& {np.round(sa_cost[_],2)} \t& {np.round(vns_ql_cost[_],2)} \t&  {gurobi_gap[_]} \t& {np.round(100*(vns_cost[_] - gurobi_cost[_])/gurobi_cost[_],3)} \t& {np.round(100*(sa_cost[_] - gurobi_cost[_])/gurobi_cost[_],3)} \t& {np.round(100*(vns_ql_cost[_] - gurobi_cost[_])/gurobi_cost[_],3)} \\\ \n')


K = [_+1 for _ in range(gurobi_data.shape[0])]
plt.figure(figsize=(10, 5))
fig, ax = plt.subplots()

plt.plot(K, gurobi_time, label = "Gurobi", marker = "*")
plt.plot(K, vns_time, label = "VNS", marker = "o")
plt.plot(K, vns_ql_time, label = "VNS-QL", marker = "s")
plt.plot(K, sa_time, label = "SA", marker = "x")
plt.xlabel("Instance", fontsize = 15,)
plt.ylabel("CPU time (Second)", fontsize = 15)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
ax.set_xticks(K)
plt.grid(visible=True, axis='both')
plt.legend()
plt.show()




stat_vns, p_vns = wilcoxon(vns_ql_time, vns_time)
stat_sa, p_sa = wilcoxon(vns_ql_time, sa_time)
print(f'QL-VNS VS VNS: {stat_vns, np.round(p_vns,4)}')
print(f'QL-VNS VS SA : {stat_sa, np.round(p_sa, 4)}')