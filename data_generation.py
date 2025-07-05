import numpy as np
import matplotlib.pyplot as plt

NB_I = 5
NB_S = 4
NB_T = 10

T = range(NB_T)
I = range(NB_I)
S = range(NB_S)

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
# E_max = np.array([np.round((1 - t * 0.1) * cap,2) if (1 - t * 0.1) * cap >= 20 else 20 for t in T])

d = np.round(10*np.array(
    [[np.random.uniform(1, 2.5) for _ in T],
     [np.random.uniform(2.5, 5) for _ in T],
     [np.random.uniform(5, 7.5) for _ in T],
     [np.random.uniform(7.5, 10) for _ in T]
    ]),2)

# for s in S:
#     d[s].sort()
# d = np.round(10*np.random.uniform(6.74, 7.84, (NB_S, NB_T)),2)

prob = np.round(np.array([1/NB_S for _ in S]),2)

# pO = np.round(np.random.uniform(110.20, 128.19, (NB_S, NB_T)),2)
pO = np.round(np.random.uniform(110.20, 128.19, (NB_S, NB_T)),2)
pC = np.round(np.random.uniform(68.88, 80.12,(NB_S, NB_T)),2)

# sold_price = np.round(np.random.uniform(75, 84, (NB_S, NB_T)),2)
sold_price = np.round(np.random.uniform(30, 40, (NB_S, NB_T)),2)
buy_price = np.round(1.4*sold_price,2)

eC = np.round(np.random.uniform(2, 4, (NB_S, NB_T)),2)
eO = np.round(np.random.uniform(-6.38, 0.53, (NB_S, NB_T)),2)
# eO = np.round(np.random.uniform(0.38, 0.53, (NB_S, NB_T)),2)

with open('data.py','w') as file :
    file.write(f"import numpy as np\n\n")
    file.write(f"NB_I = {NB_I}\n")
    file.write(f"NB_S = {NB_S}\n")
    file.write(f"NB_T = {NB_T}\n")
    file.write(f"I = range(NB_I)\n")
    file.write(f"T = range(NB_T)\n")
    file.write(f"S = range(NB_S)\n\n")

with open('data.py',"a") as file:
    file.write(f'prob = np.array ( {list(prob)} )\n')
    file.write(f'b = np.array ( {list(b)} )\n')
    # file.write(f'V = np.array ( {list(V)} )\n')
    file.write(f'E_max = np.array ( {list(E_max)} )\n')

with open('data.py',"a") as file:
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