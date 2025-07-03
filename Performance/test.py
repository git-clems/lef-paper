import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from mpl_toolkits.mplot3d import axes3d  # Needed to access get_test_data()
import seaborn as sns
from matplotlib.ticker import MaxNLocator  # Pour forcer les ticks entiers
from _VNS import*
from _VNS_QL import*


# def plot_convergence(time_history, cost_history, algorithm_name='VNS'):
"""
Generates a step plot showing the algorithm's convergence over time.

Args:
    time_history (list): List of timestamps when improvements were found.
    cost_history (list): List of best objective values found at each timestamp.
    algorithm_name (str): Name of the algorithm for the plot title.
"""


# x_ql = list(range(1, len(QL_VNS) + 1))
# x_vns = list(range(1, len(VNS_val) + 1))

plt.figure(figsize=(12, 7))

# Use a step plot for accuracy
plt.step(t_hist_vns, c_hist_vns, where='post', marker='o', linestyle='-',label=f'VNS', markersize=4, markeredgewidth=1)
plt.step(t_hist_ql, c_hist_ql, where='post', marker='o', linestyle='-',label=f'QL-VNS', markersize=4, markeredgewidth=1)

# plt.title(f'Convergence of {algorithm_name} Algorithm Over Time', fontsize=16)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Objective', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout() # Adjusts plot to prevent labels from overlapping
plt.show()

# BETA = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
# COST = [58193.87, 58141.87, 58018.8,  57672.24, 57155.54, 56586.66, 55853.22, 54556.55, 52719.55, 49985.28, 47121.9 ]
# EMISSION = [ 502.1538, 497.2395, 473.9932, 302.6818, 295.58775, 245.496, 109.2744, 56.039925, -264.4792, -294.195225, -341.465]
# XO = [39.44902835, 40.19361926, 44.34114299, 83.59590395, 86.46374984, 98.55874984, 127.95874984, 143.33270062, 221.8375, 231.795, 245.5825]

# # === Figure 1 : Émissions ===
# plt.figure(figsize=(10, 5))
# plt.plot(BETA, COST, marker='*', color='black', scalex=False, label = "Fixed carbon purchase price and β ≥ 0")
# plt.ylabel("Expexcted total cost ($)")
# plt.xlabel("beta")
# plt.xticks(BETA)
# plt.axhline(y=37416.86, color="red", label = "Carbon price sell/buy = 0")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.axhline(y=0)
# plt.show()

# Generate x-values

# Plotting
# plt.figure(figsize=(10, 5))
# plt.plot(x_ql, QL_VNS, label="QL_VNS", marker='*')
# plt.plot(x_vns, VNS_val, label="VNS", marker='*')

# plt.xlabel("Iteration")
# plt.ylabel("Objective Value")
# plt.title("QL_VNS vs VNS")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(BETA, EMISSION, marker='*', color='black', scalex=False, label = "Fixed carbon purchase price and β ≥ 0")
# plt.axhline(y=671.28, color="red", label = "carbon price sell/buy = 0")
# plt.axhline(y=211.56, color="grey", label = "Carbon cap", linestyle = "--", )
# plt.text(BETA[-1] - 0.095, 250, "Carbon cap", va='center', color="black")
# plt.ylabel("Expected GHG emission (tCO₂-eq)")
# plt.xlabel("beta")
# plt.xticks(BETA)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.axhline(y=0)
# plt.show()

# # === Figure 1 : Émissions ===
# plt.figure(figsize=(10, 5))
# plt.plot(BETA, XO, marker='*', color='black', scalex=False, label = "Fixed carbon purchase price and β ≥ 0")
# plt.axhline(y=0, color="red", label = "carbon price sell/buy = 0")
# plt.ylabel("Expected LEF-based production")
# plt.xlabel("beta")
# plt.xticks(BETA)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()