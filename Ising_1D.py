import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import time

def set_cos(A_0, N):
    ### Wraper function for J/B in the dynamic case

    A = A_0 * np.cos(np.pi * np.linspace(0, N, N) / N)
    return A;

def Hamiltonian_Ising(J, B, lattice):
    ### Function to calculate the total energy of the system.
    ### J coupling coefficient assumed for now as constant
    ### B external B-Field assumed for now as constant
    ### Lattice numpy-array with spins, for now only n=1

    if lattice.ndim != 1:
        raise TypeError("Input lattice must be 1d array")

    if J == 0 and B == 0:
        raise TypeError("At least one parameter has to be non-zero.")

        ## non-interactive case
    if J == 0:
        return -B * sum(lattice);
        ## interactive but no external field
    elif B == 0:
        lattice_shift = np.roll(lattice, 1)  ### using pbc shifting all spins by >> 1 cyclic not!  elegant use % operator
        return -J * sum(lattice * lattice_shift);
        ## general case
    elif J != 0 and B != 0:
        lattice_shift = np.roll(lattice, 1)
        return -J * sum(lattice * lattice_shift) - B * sum(lattice);

def naive_metropolis_static(Hamiltonian, J, B, configuration, n_steps, beta):
    ### Basic Metropolis approach - not optimized jet
    ### Hamiltonian - function to calculate total energy of the system
    ### J, B, coupling and external field
    ### configuration - np array with initial state
    ### n_steps - number of montecarlo moves
    ### beta - 1/kbT

    N = len(configuration)
    M = np.empty(int(n_steps / 10))
    k = 0
    for i in range(n_steps):

        flip = np.random.randint(N)
        lattice_trial = np.copy(configuration)
        lattice_trial[flip] = -1 * lattice_trial[flip]

        Del_E = Hamiltonian(J, B, lattice_trial) - Hamiltonian(J, B, configuration)
        print(Del_E)

        if Del_E <= 0:
            configuration = lattice_trial

        elif np.exp(- Del_E * beta) > np.random.uniform():
            configuration = lattice_trial

        else:
            pass

        if i % 10 == 0:
            print(f'{i} Energy = {Hamiltonian(J, B, configuration)} \n Spin = {configuration.mean()}')
            M[k] = configuration.mean()
            k = k + 1
    return M;

def quick_metropolis(Energy, J, B, configuration, n_steps, beta):
    ### Metropolis approach without calculating total energy - only energy changes
    ### Energy total energy of the input configuration
    ### J, B, coupling and external field
    ### configuration - np array with initial state
    ### n_steps - number of montecarlo moves
    ### beta - 1/kbT

    def dE(J, B, spin, spin_left, spin_right):
        delta = -2 * (B * spin + J * spin * (spin_left + spin_right))
        return delta;

    N = len(configuration)
    M = np.empty(int(n_steps / 10))
    safe = 0

    for i in range(n_steps):

        flip = np.random.randint(N)
        lattice_trial = np.copy(configuration)
        lattice_trial[flip] = -1 * lattice_trial[flip]

        '''if flip == 0:
            spin_left = lattice_trial[-1]
            spin_right = lattice_trial[flip + 1]
        elif flip == N-1:
            spin_right = lattice_trial[0]
            spin_left = lattice_trial[flip - 1]
        else :
            spin_left = lattice_trial[flip - 1]
            spin_right = lattice_trial[flip + 1]

        Del_E = dE(J, B, lattice_trial[flip], spin_left, spin_right)
        '''

        Del_E = dE(J, B, lattice_trial[flip], lattice_trial[(flip - 1) % N],
                   lattice_trial[(flip + 1) % N])  ### mod ensures pbc on flip=0 and flip=N
        print(Del_E)
        if Del_E <= 0:
            # print("case1")
            configuration = lattice_trial
            Energy += Del_E

        elif min(1, np.exp(- Del_E * beta)) > np.random.uniform():
            # print("case2")
            configuration = lattice_trial
            Energy += Del_E

        else:
            # print("reject")
            pass

        if i % 10 == 0:
            print(f'{i} Energy = {Energy} \n Spin = {configuration.mean()}')
            M[safe] = configuration.mean()
            safe += 1

    return M;

def naive_metropolis_dynamic(Hamiltonian, J, B, configuration, n_steps, beta):
    ### Basic Metropolis approach - not optimized jet
    ### Hamiltonian - function to calculate total energy of the system
    ### J, B, coupling and external field assumed as functions of time
    ### configuration - np array with initial state
    ### n_steps - number of montecarlo moves
    ### beta - 1/kbT
    skip = 100
    N = len(configuration)
    M = np.empty(int(n_steps / skip))
    k = 0
    time_line = np.empty((int(n_steps / skip), N))
    for i in range(n_steps):

        flip = np.random.randint(N)
        lattice_trial = np.copy(configuration)
        lattice_trial[flip] = -1 * lattice_trial[flip]

        Del_E = Hamiltonian(J[i], B[i], lattice_trial) - Hamiltonian(J[i], B[i], configuration)
        # print(Del_E)

        if Del_E <= 0:
            configuration = lattice_trial

        elif np.exp(- Del_E * beta) > np.random.uniform():
            configuration = lattice_trial

        else:
            pass

        if i % skip == 0:
            print(f'{i} Energy = {Hamiltonian(J[i], B[i], configuration)} \n Spin = {configuration.mean()}')
            M[k] = configuration.mean()
            time_line[k, :] = configuration
            k = k + 1

    return M, time_line;

def quick_metropolis_dynamic(Hamiltonian, J, B, configuration, n_steps, beta):
    ### Basic Metropolis approach - Energy calculation is skipped and replaced by energy difference -> quicker
    ### Hamiltonian - function to calculate total energy of the system
    ### J, B, coupling and external field assumed as functions of time -> array of length n_steps
    ### configuration - np array with initial state
    ### n_steps - number of montecarlo moves
    ### beta - 1/kbT
    skip = 100
    N = len(configuration)
    M = np.empty(int(n_steps / skip))
    k = 0
    time_line = np.empty((int(n_steps / skip), N))
    Energy = Hamiltonian(J[0], B[0], configuration)

    for i in range(n_steps):

        flip = np.random.randint(N)
        lattice_trial = np.copy(configuration)
        lattice_trial[flip] = -1 * lattice_trial[flip]

        Del_E = -2 * B[i] * lattice_trial[flip] - 2 * J[i] * lattice_trial[flip] * (
                lattice_trial[(flip - 1) % N] + lattice_trial[(flip + 1) % N])

        # print(Del_E)

        if Del_E <= 0:
            configuration = lattice_trial
            Energy += Del_E


        elif np.exp(- Del_E * beta) > np.random.uniform():
            configuration = lattice_trial
            Energy += Del_E


        else:
            pass

        if i % skip == 0:
            print(f'{i} Energy = {Energy} \n Spin = {configuration.mean()}')
            M[k] = configuration.mean()
            time_line[k, :] = configuration
            k = k + 1

    return M, time_line;

#todo: replace np.copy with single spin flip -> speed-up + less memory -> done for step metropolis
def quick_metropolis_dynamic_time(Hamiltonian, J, B, configuration, n_steps, beta):
    ### Basic Metropolis approach - Energy calculation is skipped and replaced by energy difference -> quicker
    ### Hamiltonian - function to calculate total energy of the system
    ### J, B, coupling and external field assumed as functions of time -> array of length n_steps
    ### configuration - np array with initial state
    ### n_steps - number of montecarlo moves
    ### beta - 1/kbT

    N = len(configuration)
    skip = 100
    M = np.empty(n_steps)
    k = 0
    time_line = np.empty((n_steps, N))
    Energy = Hamiltonian(J[0], B[0], configuration)
    eq_steps = int(n_steps / 2)

    for i in range(eq_steps):
        flips = np.random.choice(np.arange(0, N), N, replace=False)
        for flip in flips:
            lattice_trial = np.copy(configuration)
            lattice_trial[flip] = -1 * lattice_trial[flip]

            Del_E = -2 * B[0] * lattice_trial[flip] - 2 * J[0] * lattice_trial[flip] * (
                    lattice_trial[(flip - 1) % N] + lattice_trial[(flip + 1) % N])

            # print(Del_E)

            if Del_E <= 0:
                configuration = lattice_trial
            elif np.exp(- Del_E * beta) > np.random.uniform():
                configuration = lattice_trial
            else:
                pass

    #print(f'Spin_average of semi-stable {configuration.mean()}')


    for i in range(n_steps):

        flips = np.random.choice(np.arange(0, N), N, replace=False)

        for flip in flips:

            lattice_trial = np.copy(configuration)
            lattice_trial[flip] = -1 * lattice_trial[flip]

            Del_E = -2 * B[i] * lattice_trial[flip] - 2 * J[i] * lattice_trial[flip] * (
                    lattice_trial[(flip - 1) % N] + lattice_trial[(flip + 1) % N])
            ## https://jqgoh.github.io/ising.html modulo PBC -1 % N = N-1, always takes sign from the "divisor"

            # print(Del_E)

            if Del_E <= 0:
                configuration = lattice_trial
                Energy += Del_E


            elif np.exp(- Del_E * beta) > np.random.uniform():
                configuration = lattice_trial
                Energy += Del_E

            else:
                pass

        if i % skip == 0:
            print(f'{i} Energy = {Energy} \n Spin = {configuration.mean()} \n B = {B[i]}')
        M[i] = configuration.mean()
        time_line[i, :] = configuration


    return M, time_line;

#testing with one time unit is one mc step instead of sweeps
def step_metropolis(Hamiltonian, J, B, configuration, n_steps, beta):
    ### Basic Metropolis approach - Energy calculation is skipped and replaced by energy difference -> quicker
    ### Hamiltonian - function to calculate total energy of the system
    ### J, B, coupling and external field assumed as functions of time -> array of length n_steps
    ### configuration - np array with initial state
    ### n_steps - number of montecarlo moves
    ### beta - 1/kbT

    N = len(configuration)
    skip = 100
    M = np.empty(n_steps)
    time_line = np.empty((n_steps, N))
    eq_steps = int(n_steps / 2)

    for i in range(eq_steps):
        flip = np.random.choice(np.arange(0, N), size=1)

        lattice_trial = -configuration[flip]

        Del_E = -2 * B[0] * lattice_trial - 2 * J[0] * lattice_trial * (
                configuration[(flip - 1) % N] + configuration[(flip + 1) % N])

        if Del_E <= 0:
            configuration[flip] = lattice_trial
        elif np.exp(- Del_E * beta) > np.random.uniform():
            configuration[flip] = lattice_trial
        else:
            pass

    print(f'Spin_average of semi-stable {configuration.mean()}')
    Energy = Hamiltonian(J[0], B[0], configuration)
    print(configuration)
    for i in range(n_steps):

        flip = np.random.choice(np.arange(0, N), size=1)

        lattice_trial = -configuration[flip]

        Del_E = -2 * B[i] * lattice_trial - 2 * J[i] * lattice_trial* (
                configuration[(flip - 1) % N] +configuration[(flip + 1) % N])

        if Del_E <= 0:
            configuration[flip] = lattice_trial
            Energy += Del_E


        elif np.exp(- Del_E * beta) > np.random.uniform():
            configuration[flip] = lattice_trial
            Energy += Del_E

        else:
            pass

        if i % skip == 0:
            print(f'{i} Energy = {Energy} \n Spin = {configuration.mean()} \n B = {B[i]}')
        M[i] = configuration.mean()
        time_line[i, :] = configuration

    return M, time_line;

def reverse_step_metropolis(Hamiltonian, J, B, configuration, n_steps, beta):
    ### Basic Metropolis approach - Energy calculation is skipped and replaced by energy difference -> quicker
    ### Hamiltonian - function to calculate total energy of the system
    ### J, B, coupling and external field assumed as functions of time -> array of length n_steps
    ### configuration - np array with initial state
    ### n_steps - number of montecarlo moves
    ### beta - 1/kbT

    N = len(configuration)
    skip = 100
    M = np.empty(n_steps)
    time_line = np.empty((n_steps, N))
    eq_steps = int(n_steps / 2)

    B_r = np.flipud(B)
    J_r = np.flipud(J)

    for i in range(eq_steps):
        flip = np.random.choice(np.arange(0, N), 1, replace=False)

        lattice_trial = -configuration[flip]

        Del_E = -2 * B_r[0] * lattice_trial - 2 * J_r[0] * lattice_trial * (
                configuration[(flip - 1) % N] + configuration[(flip + 1) % N])

        if Del_E <= 0:
            configuration[flip] = lattice_trial
        elif np.exp(- Del_E * beta) > np.random.uniform():
            configuration[flip] = lattice_trial
        else:
            pass

    print(f'Spin_average of semi-stable {configuration.mean()}')
    Energy = Hamiltonian(J_r[0], B_r[0], configuration)
    k = 0

    b = B_r[0]
    j = J_r[0]

    for i in range(n_steps):

        flip = np.random.choice(np.arange(0, N), 1, replace=False)
        lattice_trial = -configuration[flip]

        Del_E = -2 * b * lattice_trial - 2 * j * lattice_trial * (
                configuration[(flip - 1) % N] + configuration[(flip + 1) % N])

        if Del_E <= 0:
            configuration[flip] = lattice_trial
            Energy += Del_E
        elif np.exp(- Del_E * beta) > np.random.uniform():
            configuration[flip] = lattice_trial
            Energy += Del_E
        else:
            pass

        if i % skip == 0:
            print(f'{i} Energy = {Energy} \n Spin = {configuration.mean()} \n B = {b}')
        M[i] = configuration.mean()
        time_line[i, :] = configuration

        if i < n_steps-1:
            b = B_r[i + 1]
            j = J_r[i + 1]

    return M, time_line;

def reverse_quick_dynamic(Hamiltonian, J, B, configuration, n_steps, beta):
    ### calculation of the time backwards trajectories by first equilibrize the input configuration and then perform the time-steps backwards
    ### Basic Metropolis approach - Energy calculation is skipped and replaced by energy difference -> quicker
    ### Hamiltonian - function to calculate total energy of the system
    ### J, B, coupling and external field assumed as functions of time
    ### configuration - np array with initial state
    ### n_steps - number of montecarlo moves
    ### beta - 1/kbT

    eq_steps = int(n_steps / 2)
    skip = 100
    N = len(configuration)
    M = np.empty(n_steps)
    k = 0
    time_line = np.empty((n_steps, N))

    ### Reverse B and J
    B_r = np.flipud(B)
    J_r = np.flipud(J)

    ###get random configuration into a semi-stable state

    for i in range(eq_steps):

        flips = np.random.choice(np.arange(0, N), N, replace=False)

        for flip in flips:

            lattice_trial = np.copy(configuration)
            lattice_trial[flip] = -1 * lattice_trial[flip]

            Del_E = -2 * B_r[0] * lattice_trial[flip] - 2 * J_r[0] * lattice_trial[flip] * (
                    lattice_trial[(flip - 1) % N] + lattice_trial[(flip + 1) % N])

            # print(Del_E)

            if Del_E <= 0:
                configuration = lattice_trial

            elif np.exp(- Del_E * beta) > np.random.uniform():
                configuration = lattice_trial




    #print(f'Spin_average of semi-stable {configuration.mean()}')


    Energy = Hamiltonian(J_r[0], B_r[0], configuration)

    ### reverse montecarlo algorithm

    for i in range(1, n_steps + 1):

        flips = np.random.choice(np.arange(0, N), N, replace=False)

        for flip in flips:

            lattice_trial = np.copy(configuration)
            lattice_trial[flip] = -1 * lattice_trial[flip]

            Del_E = -2 * B_r[i - 1] * lattice_trial[flip] - 2 * J_r[i - 1] * lattice_trial[flip] * (
                    lattice_trial[(flip - 1) % N] + lattice_trial[(flip + 1) % N])

            # print(Del_E)

            if Del_E <= 0:
                configuration = lattice_trial
                Energy += Del_E


            elif np.exp(- Del_E * beta) > np.random.uniform():
                configuration = lattice_trial
                Energy += Del_E

            else:
                pass

        if i % skip == 0:
            print(f'{i} Energy = {Energy} \n Spin = {configuration.mean()} \n B = {B_r[i-1]}')
        M[i - 1] = configuration.mean()
        time_line[i - 1, :] = configuration

    return M, time_line;

def Solve_Ising_1d(J, B, beta):
    ### analytic solution for the spins in 1d ising model. -> compare with simmulations

    Spin = np.sinh(beta * B) + np.sinh(beta * B) * np.cosh(beta * B) / np.sqrt(
        np.sinh(beta * B) * np.sinh(beta * B) + np.exp(-4 * beta * J))
    return Spin / (np.cosh(beta * B) + np.sqrt(np.sinh(beta * B) * np.sinh(beta * B) + np.exp(-4 * beta * J)));

def get_work(Field, Spin_states, state = 1):
    ### function to calculate the work for bw/fw process for both coupling and b-field
    ### pass the according spin configuration and the time dependend field as arguments
    ### outputs the work for the whole process
    def analytic_derivative(x, N):
        dx = - 20 * np.pi * np.sin( x )
        return dx;

    if state == 1:      # B
        if Spin_states.ndim > 1:
            Spin_Sum = Spin_states.sum(axis=1)  ### axis=1 "rowsum"
        else:
            Spin_Sum = Spin_states.sum()
        N = len(Field)
        x = np.linspace(0, N, N) / N
        Field_dot = np.gradient(Field, x)
        #Field_dot = analytic_derivative(x, N)
        y = Field_dot * Spin_Sum
        work = sci.trapezoid(-y, x)

        #plt.plot(x, Field_dot, c="blue")
        #plt.plot(x, Field, c="red")
        #plt.show()
        return work;
    elif state == 0:    # J
        if Spin_states.ndim > 1:
            Spin_Sum = (Spin_states * np.roll(Spin_states, 1, axis = 1)).sum(axis = 1)
        else:
            Spin_Sum = (Spin_states * np.roll(Spin_states, 1)).sum()   #check np.roll documentary

        N = len(Field)
        x = np.pi * np.linspace(0, N, N)/N
        Field_dot = np.gradient(Field, x)
        #Field_dot = analytic_derivative(x, N)
        y = Field_dot * Spin_Sum
        work = sci.trapezoid(y, dx=x[1])
        return work;

def get_work_sum(Hamiltonian, Field, constant_field,  Spin_states, state=1):
    tau = len(Field)
    Work = 0
    if state == 1:
        for i in range(tau - 1): #variable B
            #Work = (Hamiltonian(constant_field, Field[i + 1], Spin_states[i, :]) - Hamiltonian(constant_field, Field[i], Spin_states[i, :])) + Work
            Work += (-Field[i] * sum(Spin_states[i, :])) - ( -Field[i+1] * sum(Spin_states[i, :]))
    else:
        for i in range(tau - 1): #variable J
            Work = (Hamiltonian(Field[i + 1],constant_field,  Spin_states[i, :]) - Hamiltonian(Field[i], constant_field, Spin_states[i, :])) + Work
    return Work;

#todo: double check equation used-> EQ 17 is better! 21 only holds for gaussians
def calc_dF(J, B, beta, n=10):                ### eq 21 Jarzynski "Equalities and Inequalities:... "  is it the correct approach?
    def eps(beta, B, J, mode="plus"):
        term_exp = np.exp(beta * J) * np.cosh(beta * B)
        term_sqrt = np.sqrt(np.exp(2 * beta * J) * np.cosh(beta * B)**2 - 2 * np.sinh(2 * beta * J))

        if mode == "plus":
            return term_exp + term_sqrt;
        else:
            return term_exp - term_sqrt;

    dF = - np.log((eps(beta, B, J, mode="minus")**n + eps(beta, B, J)**n) / (eps(beta, B, -J, mode="minus")**n +
                                                                             eps(beta, B, -J)**n)) / beta


    return dF;

def calc_sigmoid(beta, work, dF=0):
    return 1 / (1 + np.exp(-beta * (work - dF)));



### syntax/logic testing wont show @imports
if __name__ == "__main__":
    ### Inizialize Parameters

    N = 10 ** 1
    n_steps = N * 50
    J = -1
    B = 20
    #lattice = np.random.choice([1, -1], size=N*n_steps).reshape(n_steps, N)
    lattice = np.random.choice([1, -1], size=N)

    KbT = 1
    beta = 1 / 10
    B_list = set_cos(B, n_steps)
    J_list = J * np.ones(n_steps)
    #M, trajectory = reverse_quick_dynamic(Hamiltonian_Ising, J_list, B_list, lattice, n_steps, beta)
    #M, trajectory = step_metropolis(Hamiltonian_Ising, J_list, B_list, lattice, n_steps, beta)
    #print("work=", get_work_sum(Hamiltonian_Ising, np.flipud(B_list), J, trajectory))
    #print("work=", get_work( np.flipud(B_list), trajectory, 1))
    ### Testing structures

    B_0_list = np.linspace(-2.5, 2.5, 23)
    #B_0_list = np.array([B])
    J_0 = -1
    temp = np.array([1, 1.5, 2, 2.5, 5, 7, 8, 10, 20, 30, 40, 50, 60, 70])
    beta_list = 1/ temp
    M_av = np.empty(len(temp))
    k = 0
    B_list = []
    M_av_sol = []
    Config = []

    for B_0 in B_0_list:
        #B_list.append(set_cos(B_0, n_steps))
        B_list.append((B_0 * np.ones(n_steps)))
    J = J_0 * np.ones(n_steps)
    B = 20 * np.ones(n_steps)



    for beta in beta_list:
        lattice = np.random.choice([1, -1], size=N)
        #M = quick_metropolis(Hamiltonian_Ising(J[0], B[0], lattice), J[0], B[0], lattice, n_steps, beta) ### some issues with accept/reject ?? does not converge properly
        #M, C = naive_metropolis_dynamic(Hamiltonian_Ising, J, B, lattice, n_steps, beta)
        #M, C = quick_metropolis_dynamic(Hamiltonian_Ising, J, B[0]*np.ones(n_steps), lattice, n_steps, beta)
        #M, C = quick_metropolis_dynamic_time(Hamiltonian_Ising, J, B, lattice, n_steps, beta)
        M, C = reverse_step_metropolis(Hamiltonian_Ising, J, B, lattice, n_steps, beta)
        start_time = time.time()
        #M, C = reverse_quick_dynamic(Hamiltonian_Ising, J, B, lattice, n_steps, beta)
        print(time.time() - start_time)
        Config.append(C)
        M_av[k] = M[int(len(M) / 2):].mean()
        k += 1
        M_av_sol.append(Solve_Ising_1d(J, B, beta)[0])
    off_set = 0
    # testing for variable beta vs analytic
    data = Config[-1]
    n_rows, n_cols = data.shape
    print(M_av)
    plt.scatter(beta_list, M_av, label="MC", c="red")
    plt.plot(beta_list, M_av_sol, label="Analytical")
    plt.xlabel("beta")
    plt.ylabel("<m>")
    plt.title(f'Comparison MC/Analytic average magnetization for \n J={J[-1]} B = {B[-1]} N = {N}')
    plt.legend()
    plt.show()

    '''

    # testing for variable J vs analytic
    data = Config[-1]
    n_rows, n_cols = data.shape
    print(M_av_sol)
    plt.scatter(B_0_list, M_av, label="MC", c="red")
    plt.plot(B_0_list, M_av_sol, label="Analytical")
    plt.xlabel("Coupling J")
    plt.ylabel("<m>")
    plt.title(f'Comparison MC/Analytic average magnetization for \n B={J[-1]} Beta = {beta} N = {N}')
    plt.legend()
    plt.show()
    for configuration in Config:
        plt.scatter(np.arange(0, N, 1)[configuration == 1], off_set * np.ones(sum(configuration == 1)), c="red")
        plt.scatter(np.arange(0, N, 1)[configuration == -1], off_set * np.ones(sum(configuration == -1)), c="blue")
        off_set += 1
    plt.show()
    #testing for variable B vs analytic
    data = Config[-1]
    n_rows, n_cols = data.shape
    print(M_av_sol)
    plt.scatter(B_0_list, M_av, label="MC", c="red")
    plt.plot(B_0_list, M_av_sol, label="Analytical")
    plt.xlabel("External B Field")
    plt.ylabel("<m>")
    plt.title(f'Comparison MC/Analytic average magnetization for \n J={J[-1]} Beta = {beta} N = {N}')
    plt.legend()
    plt.show()


    for i in range(n_rows):
        plt.scatter(np.arange(0, n_cols, 1)[data[i, :] == 1], i * np.ones(sum(data[i, :] == 1)) / n_cols, c="red")
        plt.scatter(np.arange(0, n_cols, 1)[data[i, :] == -1], i * np.ones(sum(data[i, :] == -1)) / n_cols, c="blue")
        plt.xlabel("Spin-Chain")
        plt.ylabel("t / T")
    plt.show()'''
