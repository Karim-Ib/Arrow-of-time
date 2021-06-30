import numpy as np
import time
from Ising_1D import set_cos, Hamiltonian_Ising, quick_metropolis_dynamic_time, reverse_quick_dynamic
from Ising_1D import reverse_step_metropolis, step_metropolis

### using exact parameters from the paper
data_size = 2 * 10**4
N_spins = 10**1
N_steps = N_spins * 50
B = 20
J = -1

B_list = set_cos(B, N_steps)
J_list = J * np.ones(N_steps)
temperatures = np.array([10, 30, 50])
Beta = np.around(1 / temperatures, decimals=2)


### B-Protocol ###
start_time = time.time()

for temp in Beta:
    for i in range(data_size):
        lattice = np.random.choice([1, -1], size=N_spins)
        #M, config_forward = quick_metropolis_dynamic_time(Hamiltonian_Ising, J_list, B_list, lattice, N_steps, temp)
        M, config_forward = step_metropolis(Hamiltonian_Ising, J_list, B_list, lattice, N_steps, temp)
        np.savetxt(f'Data_B_full/forward/B_f_{temp}_{J}_{B}_{i}.csv', config_forward, delimiter=";")

        lattice = np.random.choice([1, -1], size=N_spins)
        M, config_backwards = reverse_step_metropolis(Hamiltonian_Ising, J_list, B_list, lattice, N_steps, temp)
        np.savetxt(f'Data_B_full/backward/B_b_{temp}_{J}_{B}_{i}.csv', config_backwards, delimiter=";")




B = -1
J = 20

J_list = set_cos(J, N_steps)
B_list = B * np.ones(N_steps)


### J-Protocol ###


for temp in Beta:
    for i in range(data_size):
        lattice = np.random.choice([1, -1], size=N_spins)
        #M, config_forward = quick_metropolis_dynamic_time(Hamiltonian_Ising, J_list, B_list, lattice, N_steps, temp)
        M, config_forward = step_metropolis(Hamiltonian_Ising, J_list, B_list, lattice, N_steps, temp)
        np.savetxt(f'Data_J_full/forward/J_f_{temp}_{J}_{B}_{i}.csv', config_forward, delimiter=";")

        lattice = np.random.choice([1, -1], size=N_spins)
        #M, config_backwards = reverse_quick_dynamic(Hamiltonian_Ising, J_list, B_list, lattice, N_steps, temp)
        M, config_backwards = reverse_step_metropolis(Hamiltonian_Ising, J_list, B_list, lattice, N_steps, temp)
        np.savetxt(f'Data_J_full/backward/J_b_{temp}_{J}_{B}_{i}.csv', config_backwards, delimiter=";")

print(time.time() - start_time)