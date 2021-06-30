import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import os
import glob
from Ising_1D import Solve_Ising_1d, quick_metropolis_dynamic, naive_metropolis_dynamic, reverse_quick_dynamic, Hamiltonian_Ising
'''
#python/numpy syntax and logic tests
test = np.array([1, 2, 3])
test2 = np.copy(test)
print(test)
test2[1] = -test2[1]
print(test, test2)
print(np.random.uniform(1))

print(all(test == 1))'''


#Ising_1D mc simulation compared to analytic results
'''
N = 10 ** 1
lattice = np.random.choice([1, -1], size=N)
print(np.arange(0, N, 1)[lattice == 1])
n_steps = 10 ** 5
J = 1 * np.ones(n_steps)
B = np.linspace(-2, 2, 10)
lattice = np.random.choice([1, -1], size=N)
KbT = 0.2
beta = 1 / KbT

ind = 0

M_naive = len(B) * [None]
M_ising = len(B) * [None]

for b in B:
    #C_naive, M = naive_metropolis_dynamic(Hamiltonian_Ising, J, b * np.ones(n_steps), lattice, n_steps, beta)
    C_naive, M = quick_metropolis_dynamic(Hamiltonian_Ising, J, b * np.ones(n_steps), lattice, n_steps, beta)
    M_naive[ind] = M[int(len(M)/2):].mean()
    M_ising[ind] = Solve_Ising_1d(J[0], b, beta)
    ind += 1

plt.plot(B, M_ising)
plt.scatter(B, M_naive)
plt.show()'''


### exploring the data


#data_txt = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\Data_combined.csv', sep=';',chunksize=64000)
'''
for chunk in data_txt:
    data = chunk.to_numpy()

n_data = data.shape[0]
print(data.shape)
data_ordered = np.empty((data.shape))
#for i in range(data.shape[1]):
    #data_ordered[:, i] = np.sort(data[:, i])

### THIS IS AN ISSUE!
unique = np.unique(data, axis=1)
print(unique.shape)


'''
'''def get_array(path, batches, data_size):
    data = []
    target = []
    counter = 0
    line_counter = 0
    labels = np.ones(data_size)
    labels[:int(data_size/2)] = 0

    while True:
        with open(path) as f:
            for lines in f:
                x = lines.split(";")
                data.append(x)
                target.append(labels[line_counter])
                counter += 1
                line_counter

                if counter == batches:
                    X = np.array(data).reshape(batches, 500, 10, 1)
                    print(X)
                    y = np.array(target)
                    yield(X, y)
                    data = []
                    target = []
                    counter = 0

a = np.array([1, 1, 1, 0, 0, 1])

n_data = 3
order = np.random.random(n_data).argsort()
data = np.array(([1, 2, 3, 3.4], [4, 5, 6, 6.6], [7, 8, 9, 9.9]))
labels= np.array([1, 2, 3])
print(data[order, :])
print(data)
'''
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



print(calc_dF(20, -1, 0.1))