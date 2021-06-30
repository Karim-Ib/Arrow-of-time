import numpy as np
import matplotlib.pyplot as plt
from Ising_1D import calc_sigmoid, calc_dF
temp = 0.02
J = 20
B = -1
ind = 402

path_b = f'Data_J_full/backward/J_b_{temp}_{J}_{B}_{ind}.csv'
path_f = f'Data_J_full/forward/J_f_{temp}_{J}_{B}_{ind}.csv'


data_b = np.flipud(np.loadtxt(path_b, delimiter=";"))
data_f = (np.loadtxt(path_f, delimiter=";"))
n_rows, n_cols = data_f.shape

'''fig, axs = plt.subplots(2)
for i in range(n_rows):
    axs[0].scatter(np.arange(0, n_cols, 1)[data_f[i, :] == 1], i * np.ones(sum(data_f[i, :] == 1)) / n_rows, c="red")
    axs[0].scatter(np.arange(0, n_cols, 1)[data_f[i, :] == -1], i * np.ones(sum(data_f[i, :] == -1)) / n_rows, c="blue")
    axs[0].set_title("Forward process")
    axs[1].scatter(np.arange(0, n_cols, 1)[data_b[i, :] == 1], i * np.ones(sum(data_b[i, :] == 1)) / n_rows, c="red")
    axs[1].scatter(np.arange(0, n_cols, 1)[data_b[i, :] == -1], i * np.ones(sum(data_b[i, :] == -1)) / n_rows, c="blue")
    axs[1].set_title("Backward process")
#fig.xlabel("Spin-Chain")
#fig.ylabel("t / T")
fig.suptitle(f'Snapshot of beta = {temp}, J = {J}, B = {B}, i = {ind}')
plt.show()'''

'''
##chessboard style plot

fig, ax = plt.subplots(1, 2)
ax[0].imshow(data_f, cmap="Greys", interpolation="none", extent=[0, 20, 0, 50])
ax[1].imshow(data_b, cmap="Greys", interpolation="none", extent=[0, 20, 0, 50])
ax[0].set_yticks([0, 50])
ax[0].set_yticklabels(["0", "1"])
ax[0].set_xticks([0, 10, 20])
ax[0].set_xticklabels(["0", "5", "10"])
ax[0].set_title("Forward process")
ax[0].set_xlabel("Spin-chain")
ax[0].set_ylabel("Scaled Time")
ax[1].set_yticks([0, 50])
ax[1].set_yticklabels(["0", "1"])
ax[1].set_xticks([0, 10, 20])
ax[1].set_xticklabels(["0", "5", "10"])
ax[1].set_title("Backward process")
fig.suptitle(f'Spin configuration for Beta = {temp}')
fig.subplots_adjust(top=0.85)
plt.show()
'''






### work plots
### appear to be in wrong order??


work_fw = np.loadtxt(f'Data_J_full/work_fw.csv', delimiter=";")
work_bw = np.loadtxt(f'Data_J_full/work_bw.csv', delimiter=";")
#beta=0.1
plt.subplot(3, 1, 1)
plt.hist(work_fw[(2*int(len(work_fw)/3)):], density=True, bins=25, alpha=0.6, color="red", label="fw")
plt.hist(work_bw[(2*int(len(work_fw)/3)):], density=True, bins=25, alpha=0.6, color="blue", label="bw")

plt.title("1/beta = 10")
plt.legend()
#beta=0.02
plt.subplot(3, 1, 3)
plt.hist(work_fw[0:int(len(work_fw)/3)], density=True, bins=25, alpha=0.6, color="red", label="forward")
plt.hist(work_bw[0:int(len(work_fw)/3)], density=True, bins=25, alpha=0.6, color="blue", label="backward")
plt.title("1/beta = 50")
#beta=0.03
plt.subplot(3, 1, 2)
plt.hist(work_fw[int(len(work_fw)/3):(2*int(len(work_fw)/3))], density=True, bins=25, alpha=0.6, color="red", label="fw")
plt.hist(work_bw[int(len(work_fw)/3):(2*int(len(work_fw)/3))], density=True, bins=25, alpha=0.6, color="blue", label="bw")
plt.title("1/beta = 30")

plt.show()


###sigmoid
#beta = 0.1
w_1 = np.sort(np.concatenate([work_fw[(2*int(len(work_fw)/3)):], work_bw[(2*int(len(work_fw)/3)):]]))
#f_1 = calc_dF(w_1, 0.1)
S_1 = calc_sigmoid(0.1, w_1)
plt.subplot(3, 1, 1)
plt.plot(w_1, S_1)
plt.ylim([0,1])
plt.title("beta = 0.1")
#beta = 0.02
w_2 = np.sort(np.concatenate([work_fw[0:int(len(work_fw)/3)],work_bw[0:int(len(work_fw)/3)]]))
#f_2 = calc_dF(w_1, 0.02)
S_2 = calc_sigmoid(1/50, w_2)
plt.subplot(3, 1, 3)
plt.plot(w_2, S_2)
plt.ylim([0,1])
plt.title("beta = 0.02")
#beta = 0.03
w_3 = np.sort(np.concatenate([work_fw[int(len(work_fw)/3):(2*int(len(work_fw)/3))], work_bw[int(len(work_fw)/3):(2*int(len(work_fw)/3))]]))
#f_3 = calc_dF(w_1, 0.03)
S_3 = calc_sigmoid(1/30, w_3)
plt.subplot(3, 1,2)
plt.plot(w_3, S_3)
plt.ylim([0,1])
plt.title("beta = 0.03")
plt.xlabel("work")
plt.ylabel("p")
plt.show()