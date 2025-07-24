# from scipy.stats import wilcoxon

# # Paired Z values
# vns = [251502.08, 305935.68, 219992.60, 92278.86]
# ql_vns = [250546.73, 306262.37, 220779.15, 92524.14]

# # Run test
# stat, p = wilcoxon(ql_vns, vns)
# print(f"Wilcoxon statistic: {stat}, p-value: {p}")

import numpy as np

I = np.array([10, 50])
T = np.array([10, 30])
S = np.array([10, 50])
P = np.array([5, 10])

range_I = []
range_S = []
range_T = []
range_P = []

i = 0
compt = 1
while i < len(I):
    i_index = I[i]
    t = 0
    while t < len(T):
        t_index = T[t]
        s = 0
        while s < len(S):
            s_index = S[s]
            j = 0
            while j < len(P):
                j_index = P[j]
                # print(f"Instance {compt} ({i_index}-{t_index}-{s_index}-{j_index})")
                j += 1
                compt += 1
                range_I.append(i_index)
                range_S.append(s_index)
                range_T.append(t_index)
                range_P.append(j_index)
            s += 1
        t += 1
    i += 1

print(range_I)
print(range_S)
print(range_T)
print(range_P)