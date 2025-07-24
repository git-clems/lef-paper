import numpy as np

range_I = [2, 10, 15, 25, 50, 70, 100, 100, 100, 150, 200]
range_S = [3, 15, 20, 50, 70, 100, 120, 150, 150, 150, 200]
range_T = [4, 20, 30, 70, 100, 100, 120, 150, 170, 170, 200]
range_P = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

# range_I = [10, 10, 10, 10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 50, 50, 50]
# range_S = [10, 10, 50, 50, 10, 10, 50, 50, 10, 10, 50, 50, 10, 10, 50, 50]
# range_T = [10, 10, 10, 10, 30, 30, 30, 30, 10, 10, 10, 10, 30, 30, 30, 30]
# range_P = [5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10]

for instance in range(len(range_I)):
    
    NB_I = range_I[instance]
    NB_S = range_S[instance]
    NB_T = range_T[instance]
    NB_P = range_P[instance]

    I = range(NB_I)
    S = range(NB_S)
    T = range(NB_T)
    P = range(NB_P)

    b = np.round(np.random.uniform(5.0, 7.0, (NB_I)), 2)
    # V = np.round(np.random.uniform(98, 353, (NB_T)), 2)

    initial_cap = 45*NB_T/10
    cap = initial_cap
    # E_max = np.array([cap for _ in T])
    E_max = [initial_cap]

    for t in T[1:]:
        a = 0.2
        goal = 0.25*initial_cap
        cap -= a*cap
        if cap > goal:
            E_max.append(np.round(cap,2))
        else:
            E_max.append(goal)

    d = np.round(10*np.random.uniform(6.74, 7.84, (NB_S, NB_T, NB_P)),2)

    prob = np.round(np.array([1/NB_S for _ in S]),2)

    # pO = np.round(np.random.uniform(110.20, 128.19, (NB_S, NB_T, NB_P)),2)
    pO = np.round(np.random.uniform(110.20, 128.19, (NB_S, NB_T, NB_P)),2)
    pC = np.round(np.random.uniform(68.88, 80.12,(NB_S, NB_T, NB_P)),2)

    # sold_price = np.round(np.random.uniform(75, 84, (NB_S, NB_T, NB_P)),2)
    sold_price = np.round(np.random.uniform(30, 40, (NB_S, NB_T)),2)
    buy_price = np.round(1.4*sold_price,2)

    eC = np.round(np.random.uniform(0.91, 1.65, (NB_S, NB_T, NB_P)),2)
    eO = np.round(np.random.uniform(-6.38, 0.53, (NB_S, NB_T, NB_P)),2)
    # eO = np.round(np.random.uniform(0.38, 0.53, (NB_S, NB_T, NB_P)),2)

    with open(f'Performance/Data/instance_{instance+1}.py','w') as file :
        file.write(f"import numpy as np\n\n")
        file.write(f"NB_I = {NB_I}\n")
        file.write(f"NB_S = {NB_S}\n")
        file.write(f"NB_T = {NB_T}\n")
        file.write(f"NB_P = {NB_P}\n")
        file.write(f"I = range(NB_I)\n")
        file.write(f"T = range(NB_T)\n")
        file.write(f"S = range(NB_S)\n")
        file.write(f"P = range(NB_P)\n\n")

    with open(f'Performance/Data/instance_{instance+1}.py',"a") as file:
        file.write(f'prob = np.array ( {list(prob)} )\n')
        file.write(f'b = np.array ( {list(b)} )\n')
        # file.write(f'V = np.array ( {list(V)} )\n')
        file.write(f'E_max = np.array ( {list(E_max)} )\n')

    with open(f'Performance/Data/instance_{instance+1}.py',"a") as file:
        file.write('d = np.array ( [\n')
        for k in S:
            file.write(f'\t{list(d[k])},\n')
        file.write('] )\n')
        
        file.write('eC = np.array ( [\n')
        for k in S:
            file.write(f'\t{list(eC[k])},\n')
        file.write('] )\n')
        
        file.write('eO = np.array ( [\n')
        for k in S:
            file.write(f'\t{list(eO[k])},\n')
        file.write('] )\n')

        file.write('pO = np.array ( [\n')
        for k in S:
            file.write(f'\t{list(pO[k])},\n')
        file.write('] )\n')
        
        file.write('pC = np.array ( [\n')
        for k in S:
            file.write(f'\t{list(pC[k])},\n')
        file.write('] )\n')  

        file.write('sold_price = np.array ( [\n')
        for k in S:
            file.write(f'\t{list(sold_price[k])},\n')
        file.write('] )\n')
        file.write('buy_price = np.array ( [\n')
        for k in S:
            file.write(f'\t{list(buy_price[k])},\n')
        file.write('] )\n')