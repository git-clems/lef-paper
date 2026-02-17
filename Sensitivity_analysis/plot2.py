import matplotlib.pyplot as plt
import numpy as np
from Sensitivity_analysis.data import R

r = 100*np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15])


ox = [106.32, 106.32, 89.19, 85.88, 27.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ]
cost = [57444.57, 59504.14, 61402.88, 62937.67, 63869.39, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, 64191.59, ]
emi = [329.13, 329.13, 395.36, 408.6, 658.96, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, 783.89, ]



install = [4119.14, 6178.7, 6569.6, 7796.54, 2703.18, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ]
prod = [[49731.21], [49731.21], [49044.15], [48911.1], [46509.41], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], [45353.12], ]
trade = [3594.23, 3594.23, 5789.12, 6230.04, 14656.79, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, 18838.46, ]

plt.figure(figsize=(7,6))
plt.plot(r[:8], install[:8], label = 'LEF investment', marker = 'x')
plt.plot(r[:8], prod[:8], label = 'Production cost', marker = 'o')
plt.plot(r[:8], trade[:8], label = 'Carbon cost', marker = 's')
plt.xlabel(xlabel='Maintenance cost compared to installation cost (%)', fontsize=18)
plt.ylabel(ylabel='Cost', fontsize=18)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize=18)
plt.grid(visible=True)
plt.show()

plt.figure(figsize=(7,6))
plt.plot(r[:8], emi[:8], label = 'Emission', marker = 'o')
plt.xlabel(xlabel='Maintenance cost compared to installation cost (%)', fontsize=18)
plt.ylabel(ylabel='Total expected emission', fontsize=18)
plt.legend(fontsize=18)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid(visible=True)
plt.show()


plt.figure(figsize=(7,6))
plt.plot(r[:8], ox[:8], label = 'LEF-based production', marker = 'o')
plt.xlabel(xlabel='Maintenance cost compared to installation cost (%)', fontsize=18)
plt.ylabel(ylabel='Total LEF-based production', fontsize=18)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize=18)
plt.grid(visible=True)
plt.show()


# fig, ax = plt.subplots()

# l1, = ax.plot(r, emi, label='Emission', marker='o', c='red')
# ax.set_xlabel('Maintenance cost compared to installation cost (%)')
# ax.set_ylabel('Total expected emission', c='red')

# ax2 = ax.twinx()
# l2, = ax2.plot(r, cost, label='Cost', marker='x', color='blue')
# ax2.set_ylabel('Total expected cost', c = 'blue')

# # Fusion des légendes
# ax.legend(handles=[l1, l2], loc='best')

# ax.grid(True)
# plt.show()