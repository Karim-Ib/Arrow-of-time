import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Ising_1D import get_work, calc_sigmoid, set_cos, calc_dF

model = tf.keras.models.load_model(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\CNN_TF')
#load test data
temp = "med"
t = 1
if temp == "cold":
    t = 1/10
elif temp == "med":
    t = np.around(1/30, 2)
elif temp == "hot":
    t = 1/50

print(t)

x_test = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\Data_{}.csv'.format(temp), sep=';')
y_test= pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\labels_{}.csv'.format(temp), sep=';')
x = x_test.to_numpy()
y = y_test.to_numpy()
n_data = x.shape[0]
data = np.copy(x)

data[:int(n_data/2)] = np.fliplr(data[:int(n_data/2)])
data = data.reshape(-1, 500, 10, 1)

#predict
score = model.evaluate(data *t, y, verbose=1)


#work plot
p = model.predict(data*t)
y = y.reshape(-1)

work = np.ones(len(y))
dF = work.copy()
for i in range(int(n_data/2)):
    work[i] = -get_work(set_cos(20, 500), x[i, :].reshape(500, 10), 0)
for i in range(int(n_data/2), n_data):
    work[i] = get_work(np.flipud(set_cos(20, 500)), np.flipud(x[i, :].reshape(500, 10)), 0)


#sigmoid = calc_sigmoid(t, np.arange(-100, 100, 1))

plt.scatter(work[list(*np.where(y == 0))], 1-p[list(*np.where(y == 0)), 0], c="grey", marker="o", label="Logistic output")
plt.scatter(work[list(*np.where(y == 1))], 1-p[list(*np.where(y == 1)), 0], c="grey", marker="o")
plt.plot(np.arange(-100, 100, 1), calc_sigmoid(t, np.arange(-100, 100, 1), calc_dF(20, -1, t)), c="black", lw=4, label="analytic likelihood")
print(calc_dF(20, -1, t))
plt.text(50, 0.2, f'Accuracy {np.around(score[1], 3)}')
plt.xlabel("work")
plt.ylabel("p")
plt.ylim(0, 1)
#plt.title("CNN Results for beta = 1/50")
plt.title(f"Logit Result for beta = {t}")
plt.legend()
plt.show()
