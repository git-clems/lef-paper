import numpy as np

NB_I = 2
NB_S = 2
NB_T = 4
NB_P = 1
I = range(NB_I)
T = range(NB_T)
S = range(NB_S)
P = range(NB_P)
V = 251.7
u = 25.17

prob = np.array ( [np.float64(0.5), np.float64(0.5)] )
b = np.array ( [np.float64(3.1), np.float64(3.43)] )
E_max = np.array ( [18.0, np.float64(14.4), np.float64(11.52), np.float64(9.22)] )
d = np.array ( [
	[[59.04], [48.13], [68.13], [49.38]],
	[[90.26], [96.73], [44.51], [81.26]],
] )
eC = np.array ( [
	[[0.95], [1.4], [1.41], [1.07]],
	[[1.01], [1.14], [1.18], [1.33]],
] )
eO = np.array ( [
	[[-3.35], [0.45], [-5.67], [-4.94]],
	[[-5.27], [-1.87], [-4.63], [-3.16]],
] )
pO = np.array ( [
	[[119.71], [120.42], [126.85], [111.48]],
	[[111.77], [110.56], [125.18], [124.2]],
] )
pC = np.array ( [
	[[78.66], [79.88], [77.86], [74.07]],
	[[77.65], [70.21], [76.07], [70.49]],
] )
sold_price = np.array ( [
	[np.float64(39.45), np.float64(35.22), np.float64(34.15), np.float64(32.65)],
	[np.float64(37.74), np.float64(34.56), np.float64(35.68), np.float64(30.19)],
] )
buy_price = np.array ( [
	[np.float64(36.18), np.float64(36.12), np.float64(36.17), np.float64(39.44)],
	[np.float64(36.82), np.float64(33.6), np.float64(34.37), np.float64(36.98)],
] )
