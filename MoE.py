import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

gate = tf.keras.models.load_model(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\gate')
data_B_cold = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\Data_cold.csv', sep=';')
data_B_med = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\Data_med.csv', sep=';')
data_B_hot = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\Data_hot.csv', sep=';')
data_J_cold = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\Data_cold.csv', sep=';')
data_J_med = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\Data_med.csv', sep=';')
data_J_hot = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\Data_hot.csv', sep=';')
data_B_cold = data_B_cold.to_numpy()
data_B_med = data_B_med.to_numpy()
data_B_hot = data_B_hot.to_numpy()
data_J_cold = data_J_cold.to_numpy()
data_J_med = data_J_med.to_numpy()
data_J_hot = data_J_hot.to_numpy()
data_cold = np.concatenate((data_B_cold, data_J_cold), axis=0)
data_med = np.concatenate((data_B_med, data_J_med), axis=0)
data_hot = np.concatenate((data_B_hot, data_J_hot), axis=0)

labels = np.zeros(data_cold.shape[0])
labels[int(data_cold.shape[0] / 2):] = 1
print(data_cold.shape, labels.shape)
data_cold = np.array(data_cold).reshape(-1, 500, 10, 1)
data_med = np.array(data_med).reshape(-1, 500, 10, 1)
data_hot = np.array(data_hot).reshape(-1, 500, 10, 1)

x_train, x_test_cold, y_train, y_test_cold = train_test_split(data_cold, labels, test_size=0.2)
x_train, x_test_med, y_train, y_test_med = train_test_split(data_med, labels, test_size=0.2)
x_train, x_test_hot, y_train, y_test_hot = train_test_split(data_hot, labels, test_size=0.2)

p_cold = gate.predict(x_test_cold)
p_med = gate.predict(x_test_med)
p_hot = gate.predict(x_test_hot)

plt.subplot(1, 3, 1)
plt.scatter(np.where(y_test_cold == 0), p_cold[list(*np.where(y_test_cold == 0)), 0], c="blue", marker="o", label="J")
plt.scatter(np.where(y_test_cold == 1), p_cold[list(*np.where(y_test_cold == 1)), 0], c="red", marker="o", label="B")
plt.ylabel("p")
plt.ylim(0, 1)
plt.title("cold")
plt.legend()
plt.subplot(1, 3, 2)
plt.scatter(np.where(y_test_med == 0), p_med[list(*np.where(y_test_med == 0)), 0], c="blue", marker="o", label="J")
plt.scatter(np.where(y_test_med == 1), p_med[list(*np.where(y_test_med == 1)), 0], c="red", marker="o", label="B")
plt.ylim(0, 1)
plt.title("med")
plt.subplot(1, 3, 3)
plt.scatter(np.where(y_test_hot == 0), p_hot[list(*np.where(y_test_hot == 0)), 0], c="blue", marker="o", label="J")
plt.scatter(np.where(y_test_hot == 1), p_hot[list(*np.where(y_test_hot == 1)), 0], c="red", marker="o", label="B")
plt.ylim(0, 1)
plt.title("hot")
plt.show()


CNN = tf.keras.models.load_model(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\binary_clf')

