import numpy as np

NB_I = 10
NB_S = 4
NB_T = 10
I = range(NB_I)
T = range(NB_T)
S = range(NB_S)

prob = np.array ( [np.float64(0.25), np.float64(0.25), np.float64(0.25), np.float64(0.25)] )
b = np.array ( [np.float64(6.1), np.float64(6.43), np.float64(6.21), np.float64(6.09), np.float64(5.85), np.float64(6.29), np.float64(5.88), np.float64(6.78), np.float64(6.93), np.float64(5.77)] )
E_max = np.array ( [45, np.float64(36.0), np.float64(28.8), np.float64(23.04), np.float64(18.43), np.float64(14.75), np.float64(11.8), 11.25, 11.25, 11.25] )
d = np.array ( [
	[np.float64(21.88), np.float64(17.93), np.float64(18.52), np.float64(23.88), np.float64(11.07), np.float64(11.31), np.float64(10.3), np.float64(22.49), np.float64(21.67), np.float64(23.05)],
	[np.float64(49.47), np.float64(44.98), np.float64(36.54), np.float64(44.51), np.float64(27.96), np.float64(41.0), np.float64(28.58), np.float64(48.62), np.float64(38.05), np.float64(35.37)],
	[np.float64(56.61), np.float64(69.36), np.float64(61.4), np.float64(64.21), np.float64(50.47), np.float64(65.44), np.float64(65.3), np.float64(65.42), np.float64(73.59), np.float64(67.05)],
	[np.float64(83.99), np.float64(85.93), np.float64(92.44), np.float64(76.51), np.float64(91.67), np.float64(91.77), np.float64(80.26), np.float64(78.22), np.float64(82.89), np.float64(84.09)],
] )
eC = np.array ( [
	[np.float64(1.3), np.float64(0.95), np.float64(1.06), np.float64(0.92), np.float64(1.5), np.float64(1.08), np.float64(1.17), np.float64(1.6), np.float64(1.43), np.float64(0.93)],
	[np.float64(1.03), np.float64(1.37), np.float64(1.34), np.float64(1.09), np.float64(1.6), np.float64(1.36), np.float64(1.31), np.float64(1.35), np.float64(1.45), np.float64(1.14)],
	[np.float64(1.2), np.float64(1.07), np.float64(1.05), np.float64(1.61), np.float64(1.46), np.float64(1.27), np.float64(1.08), np.float64(1.1), np.float64(0.95), np.float64(1.23)],
	[np.float64(1.14), np.float64(1.43), np.float64(1.19), np.float64(1.04), np.float64(0.93), np.float64(0.96), np.float64(1.41), np.float64(1.25), np.float64(1.31), np.float64(1.57)],
] )
eO = np.array ( [
	[np.float64(0.46), np.float64(-4.88), np.float64(-1.8), np.float64(-4.56), np.float64(-6.24), np.float64(-1.14), np.float64(-4.17), np.float64(-3.73), np.float64(-2.31), np.float64(-0.64)],
	[np.float64(-2.03), np.float64(-0.35), np.float64(-4.49), np.float64(-0.87), np.float64(-5.1), np.float64(0.2), np.float64(-1.63), np.float64(-4.89), np.float64(0.17), np.float64(-1.33)],
	[np.float64(-4.63), np.float64(-4.91), np.float64(-2.8), np.float64(-6.2), np.float64(-4.95), np.float64(-3.45), np.float64(-3.79), np.float64(-3.18), np.float64(-4.46), np.float64(-2.33)],
	[np.float64(-0.41), np.float64(-5.57), np.float64(-2.8), np.float64(-5.47), np.float64(-1.43), np.float64(-3.64), np.float64(-2.47), np.float64(-5.11), np.float64(-5.38), np.float64(-3.01)],
] )
pO = np.array ( [
	[np.float64(120.46), np.float64(118.09), np.float64(127.98), np.float64(112.04), np.float64(113.96), np.float64(113.1), np.float64(121.95), np.float64(114.76), np.float64(118.59), np.float64(114.6)],
	[np.float64(113.06), np.float64(112.19), np.float64(122.01), np.float64(112.69), np.float64(113.74), np.float64(116.83), np.float64(124.97), np.float64(111.95), np.float64(125.27), np.float64(111.93)],
	[np.float64(127.77), np.float64(118.63), np.float64(127.77), np.float64(121.08), np.float64(123.5), np.float64(110.9), np.float64(115.29), np.float64(112.36), np.float64(115.53), np.float64(112.34)],
	[np.float64(115.92), np.float64(117.65), np.float64(111.35), np.float64(122.66), np.float64(120.39), np.float64(114.97), np.float64(119.61), np.float64(111.89), np.float64(120.56), np.float64(126.92)],
] )
pC = np.array ( [
	[np.float64(72.46), np.float64(76.38), np.float64(70.36), np.float64(76.93), np.float64(72.13), np.float64(70.94), np.float64(75.47), np.float64(69.11), np.float64(78.2), np.float64(68.93)],
	[np.float64(76.5), np.float64(71.91), np.float64(77.14), np.float64(79.69), np.float64(71.68), np.float64(75.36), np.float64(75.53), np.float64(75.31), np.float64(71.39), np.float64(79.59)],
	[np.float64(73.91), np.float64(78.39), np.float64(76.74), np.float64(72.22), np.float64(78.03), np.float64(73.34), np.float64(78.78), np.float64(75.41), np.float64(78.79), np.float64(76.66)],
	[np.float64(77.03), np.float64(74.51), np.float64(79.63), np.float64(76.12), np.float64(73.64), np.float64(75.7), np.float64(69.1), np.float64(72.27), np.float64(76.3), np.float64(72.14)],
] )
sold_price = np.array ( [
	[np.float64(36.18), np.float64(34.29), np.float64(31.35), np.float64(32.98), np.float64(35.7), np.float64(35.91), np.float64(35.74), np.float64(36.53), np.float64(36.52), np.float64(34.31)],
	[np.float64(38.97), np.float64(33.68), np.float64(34.36), np.float64(38.92), np.float64(38.06), np.float64(37.04), np.float64(31.0), np.float64(39.19), np.float64(37.14), np.float64(39.99)],
	[np.float64(31.49), np.float64(38.68), np.float64(31.62), np.float64(36.16), np.float64(31.24), np.float64(38.48), np.float64(38.07), np.float64(35.69), np.float64(34.07), np.float64(30.69)],
	[np.float64(36.97), np.float64(34.54), np.float64(37.22), np.float64(38.66), np.float64(39.76), np.float64(38.56), np.float64(30.12), np.float64(33.6), np.float64(37.3), np.float64(31.72)],
] )
buy_price = np.array ( [
	[np.float64(43.42), np.float64(41.15), np.float64(37.62), np.float64(39.58), np.float64(42.84), np.float64(43.09), np.float64(42.89), np.float64(43.84), np.float64(43.82), np.float64(41.17)],
	[np.float64(46.76), np.float64(40.42), np.float64(41.23), np.float64(46.7), np.float64(45.67), np.float64(44.45), np.float64(37.2), np.float64(47.03), np.float64(44.57), np.float64(47.99)],
	[np.float64(37.79), np.float64(46.42), np.float64(37.94), np.float64(43.39), np.float64(37.49), np.float64(46.18), np.float64(45.68), np.float64(42.83), np.float64(40.88), np.float64(36.83)],
	[np.float64(44.36), np.float64(41.45), np.float64(44.66), np.float64(46.39), np.float64(47.71), np.float64(46.27), np.float64(36.14), np.float64(40.32), np.float64(44.76), np.float64(38.06)],
] )
