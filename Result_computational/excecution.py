import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

gurobi_data = pd.read_excel('Result_computational/full_result.xlsx', sheet_name='gurobi', index_col='Instance')
vns_data = pd.read_excel('Result_computational/full_result.xlsx', sheet_name='vns', index_col='Instance')
vns_ql_data = pd.read_excel('Result_computational/full_result.xlsx', sheet_name='vns-ql', index_col='Instance')
as_data = pd.read_excel('Result_computational/full_result.xlsx', sheet_name='as', index_col='Instance')
as_ql_data = pd.read_excel('Result_computational/full_result.xlsx', sheet_name='as-ql', index_col='Instance')


gurobi_time = np.array(gurobi_data.iloc[:, 1:10], dtype=float).mean(axis=1)
gurobi_cost = np.array(gurobi_data['cost'], dtype=float)
gurobi_gap = np.array(gurobi_data['gap'])

vns_time = np.array(vns_data.iloc[:, 1:10].mean(axis=1), dtype=float)
vns_cost = np.array(vns_data.iloc[:,10:].mean(axis=1), dtype=float)

vns_ql_time = np.array(vns_ql_data.iloc[:, 1:10].mean(axis=1), dtype=float)
vns_ql_cost = np.array(vns_ql_data.iloc[:,10:].mean(axis=1), dtype=float)

as_time = np.array(as_data.iloc[:, 1:10].mean(axis=1), dtype=float)
as_cost = np.array(as_data.iloc[:,10:].mean(axis=1), dtype=float)

as_ql_time = np.array(as_ql_data.iloc[:, 1:10].mean(axis=1), dtype=float)
as_ql_cost = np.array(as_ql_data.iloc[:,10:].mean(axis=1), dtype=float)

with open("Result_computational/avg_result.csv","w") as file:
    file.write(f'Gurobi \t VNS \t VNS-QL \t AS \t AS-QL \t Gurobi \t VNS \t VNS-QL \t AS \t AS-QL \t Gurobi \t VNS \t VNS-QL \t AS \t AS-QL\n')
    for _ in range(gurobi_data.shape[0]):
        file.write(f'{np.round(gurobi_cost[_],3)} \t {np.round(vns_cost[_],3)} \t {np.round(vns_ql_cost[_],3)} \t {np.round(as_cost[_],3)} \t {np.round(as_ql_cost[_],3)} \t {gurobi_gap[_]} \t {np.round(100*(vns_cost[_] - gurobi_cost[_])/gurobi_cost[_],3)} \t {np.round(100*(vns_ql_cost[_] - gurobi_cost[_])/gurobi_cost[_],3)} \t {np.round(100*(as_cost[_] - gurobi_cost[_])/gurobi_cost[_],3)} \t {np.round(100*(as_ql_cost[_] - gurobi_cost[_])/gurobi_cost[_],3)} \t {np.round(gurobi_time[_],3)} \t {np.round(vns_time[_],3)} \t {np.round(vns_ql_time[_],3)} \t {np.round(as_time[_],3)} \t {np.round(as_ql_time[_],3)}\n')

print(vns_data)
plt.plot(gurobi_time, label = "Gurobi", marker = "*")
plt.xlabel("Instance")
plt.ylabel("CPU time (Second)")
plt.legend()
# plt.show()


print(type(vns_time))

# import numpy as np
# # Example with numeric strings
# string_array = ["1.1", "2.5", "3.14"]
# float_array = np.array(string_array, dtype=float)
# print(string_array) # Output: [1.1 2.5 3.14]
# print(float_array) # Output: [1.1 2.5 3.14]