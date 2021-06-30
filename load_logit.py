import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Ising_1D import get_work, calc_sigmoid, set_cos


temp = "cold"
model = tf.keras.models.load_model(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\logit_TF')
data_txt = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\Data_{}.csv'.format(temp), sep=';')
labels_txt = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\labels_{}.csv'.format(temp), sep=';')
data = data_txt.to_numpy()
labels = labels_txt.to_numpy()

x = np.copy(data)
n_data = x.shape[0]
x[int(n_data/2):] = np.fliplr(x[int(n_data/2):])
labels = labels.reshape(-1)
p = model.predict(x)

model.evaluate(x, labels)


work = np.ones(len(labels))
for i in list(*np.where(labels == 0)):
    work[i] = get_work(set_cos(20, 500), data[i, :].reshape(500, 10), 1)
for i in list(*np.where(labels == 1)):
    work[i] = -get_work(np.flipud(set_cos(20, 500)), np.flipud(data[i, :].reshape(500, 10)), 1)

sigmoid = calc_sigmoid(1/10, np.sort(work))

plt.scatter(work[list(*np.where(labels == 0))], p[list(*np.where(labels == 0))], c="blue", marker="o", label="forward")
plt.scatter(work[list(*np.where(labels == 1))], p[list(*np.where(labels == 1))], c="red", marker="o", label="backward")
plt.plot(np.sort(work), sigmoid, c="black", lw=6, label="analytic Likelihood")
plt.xlabel("work")
plt.ylabel("p")
plt.ylim(0, 1)
#plt.title("CNN Results for beta = 1/50")
plt.title("Results for Cold trained model for Med data")
plt.legend()
plt.show()
