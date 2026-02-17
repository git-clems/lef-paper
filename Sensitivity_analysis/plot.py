import matplotlib.pyplot as plt
import numpy as np
from Sensitivity_analysis.data import R

R = 100*R
cost_0 = [68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, 68981.98, ]
cost_10 = [65371.28 ,66091.56 ,66642.3 ,67056.13 ,67297.4 ,67442.99 ,67539.71 ,67606.99 ,67656.87 ,67696.44 ,67733.83 ,67752.3 ,67770.77 ,67789.24 ,67807.71 ,67826.18 ,67826.18 ,67826.18 ,67826.18 ,67826.18 ,67826.18]
cost_30 = [58158.17, 60311.86, 61961.11, 63204.07, 63928.96, 64363.9, 64655.18, 64856.63, 65006.63, 65124.99, 65237.53, 65292.94, 65348.35, 65403.76, 65459.17, 65514.58, 65514.58, 65514.58, 65514.58, 65514.58, 65514.58, ]
cost_45 = [52772.99, 55989.25, 58454.33, 60315.68, 61402.88, 62055.82, 62491.71, 62794.39, 63019.03, 63196.39, 63365.38, 63448.49, 63531.61, 63614.72, 63697.01, 63780.95, 63780.95, 63780.95, 63780.95, 63780.95, 63780.95, ]
cost_70 = [43837.22, 48818.06, 52635.44, 55510.85, 57197.14, 58211.22, 58890.02, 59359.22, 59708.24, 59983.3, 60245.99, 60374.77, 60504.02, 60633.31, 60762.6, 60891.89, 60891.89, 60891.89, 60891.89, 60891.89, 60891.89, ]
cost_150 = [15378.78, 26014.3, 34142.21, 40264.38, 43833.17, 45975.4, 47416.39, 48416.71, 49157.76, 49741.04, 50297.48, 50568.28, 50839.09, 51112.16, 51388.5, 51664.46, 51664.46, 51664.46, 51664.46, 51664.46, 51664.46, ]

no_policy_cost, no_policy_emi = 45353.12, 783.89 

plt.figure(figsize=(7,6))
plt.plot(R, cost_0, label = '0', marker = 'o')
plt.plot(R, cost_10, label = '10', marker = 'o')
plt.plot(R, cost_30, label = '30', marker = 'o')
# plt.plot(R, cost_45, label = '45', marker = 'o')
plt.plot(R, cost_70, label = '70', marker = 'o')
plt.plot(R, cost_150, label = '150', marker = 'o')
plt.axhline(
    y=no_policy_cost, 
    label = 'No carbon policy',
    color='black',
    linestyle='--',
    linewidth=2
    )
plt.xlabel(xlabel='Periodic abatement (%)', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylabel(ylabel='Total cost', fontsize = 15)
plt.legend(title = 'Initial emission', fontsize = 15)
plt.grid(visible=True)
plt.show()

emi_0 = [395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, ]
emi_10 = [395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, ]
emi_30 = [395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, 395.37, ]
emi_45 = [400.98, 400.63, 395.37, 395.36, 395.36, 395.36, 395.36, 395.36, 395.36, 395.36, 395.36, 395.36, 395.36, 395.36, 395.47, 395.36, 395.36, 395.36, 395.36, 395.36, 395.36, ]
emi_70 = [401.21, 401.09, 401.09, 395.36, 395.36, 395.36, 395.36, 395.47, 395.47, 395.47, 395.47, 395.47, 395.47, 395.47, 395.47, 395.47, 395.47, 395.47, 395.47, 395.47, 395.47, ]
emi_150 = [401.21, 401.21, 401.21, 401.09, 401.09, 395.47, 395.47, 395.47, 395.47, 401.09, 395.47, 395.47, 395.47, 395.47, 400.98, 400.98, 400.98, 400.98, 400.98, 400.98, 400.98, ]

fig, ax1 = plt.subplots(figsize=(7, 6))

# First axis: emissions by policy
ax1.plot(R, emi_0, label='0', marker='o')
ax1.plot(R, emi_10, label='10', marker='o')
ax1.plot(R, emi_30, label='30', marker='o')
# ax1.plot(R, emi_45, label='45', marker='o')
ax1.plot(R, emi_70, label='70', marker='o')
ax1.plot(R, emi_150, label='150', marker='o')

ax1.set_xlabel('Periodic abatement (%)', fontsize=15)
ax1.set_ylabel('Total expected emission', fontsize=15)
ax1.tick_params(axis='both', labelsize=15)
ax1.grid(True)

# Second axis: constant line
ax2 = ax1.twinx()
ax2.axhline(
    y=no_policy_emi,
    color='black',
    linestyle='--',
    linewidth=2,
    label='No carbon policy'
)

ax2.set_ylabel('No-policy emission', fontsize=15)
ax2.tick_params(axis='y', labelsize=15)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    title='Initial emission',
    fontsize=15
)

plt.show()
