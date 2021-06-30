import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import pandas as pd
from Ising_1D import calc_sigmoid, get_work, set_cos

#load train data

data_txt = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_full\Data.csv', sep=';')
labels_txt = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_full\Data_labels.csv', sep=';')

x = data_txt.to_numpy()
y = labels_txt.to_numpy()
n_data = x.shape[0]
n_features = x.shape[1]

#x[x == -1] = 0

#reverse backwards trajectories
x[:int(n_data/2), :] = np.fliplr(x[:int(n_data/2), :])

#scale data
x[:int(n_data / 6), :] *= 1/50
x[int(n_data / 6):(2 * int(n_data / 6)), :] *= np.around(1/30, 2)
x[(2 * int(n_data / 6)):(3 * int(n_data / 6)), :] *= 1/10
x[(3 * int(n_data / 6)):(4 * int(n_data / 6)), :] *= 1/50
x[(4 * int(n_data / 6)):(5 * int(n_data / 6)), :] *= np.around(1/30, 2)
x[(5 * int(n_data / 6)):, :] *= 1/10


#shuffle training data
order = np.random.random(n_data).argsort()
x = x[order, :]
y = y[order]
x = x.reshape(-1, 500, 10, 1)

#define model
clf = Sequential()
clf.add(Conv2D(4, (2,2), strides=1,padding="valid", activation="relu", kernel_regularizer=l2(10**-4), input_shape=(500, 10, 1)))
clf.add(Dropout(0.25))
clf.add(Flatten())
#clf.add(Dense(50, activation="relu", kernel_regularizer=l2(10**-4)))
clf.add(Dense(1, activation="sigmoid"))

#compile model
clf.summary()
optimizer_lr = tf.keras.optimizers.Adam(learning_rate=0.001)
clf.compile(optimizer=optimizer_lr, loss="binary_crossentropy", metrics=["accuracy"])

#fit model
history = clf.fit(x, y, epochs=15, verbose=1, validation_split=0.2)

#visualize
clf.save(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\CNN_TF')

plt.subplot(1, 2, 1)
plt.plot(history.epoch, history.history['loss'], label="loss" )
plt.plot(history.epoch, history.history['val_loss'], c="red", label="val_loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.title("Comparison Train/Validation-Set loss on epoch")
plt.subplot(1, 2, 2)
plt.plot(history.epoch, history.history['accuracy'], label="acc" )
plt.plot(history.epoch, history.history['val_accuracy'], c="red", label="val_acc")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.title("Comparison Train/Validation-Set accuracy on epoch")
plt.show()


#load test data
temp = "cold"
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
score = clf.evaluate(data *t, y, verbose=1)


#work plot
p = clf.predict(data*t)
print(p, p.shape)
y = y.reshape(-1)

work = np.ones(len(y))

for i in range(int(n_data/2)):
    work[i] = -get_work(set_cos(20, 500), x[i, :].reshape(500, 10), 0)
for i in range(int(n_data/2), n_data):
    work[i] = get_work(np.flipud(set_cos(20, 500)), np.flipud(x[i, :].reshape(500, 10)), 0)

#sigmoid = calc_sigmoid(t, np.arange(-100, 100, 1))

plt.scatter(work[list(*np.where(y == 0))], 1-p[list(*np.where(y == 0)), 0], c="grey", marker="o", label="Logistic output")
plt.scatter(work[list(*np.where(y == 1))], 1-p[list(*np.where(y == 1)), 0], c="grey", marker="o")
plt.plot(np.arange(-100, 100, 1), calc_sigmoid(t, np.arange(-100, 100, 1)), c="black", lw=4, label="analytic likelihood")
plt.text(50, 0.2, f'Accuracy {np.around(score[1], 3)}')
plt.xlabel("work")
plt.ylabel("p")
plt.ylim(0, 1)
#plt.title("CNN Results for beta = 1/50")
plt.title(f"Logit Result for beta = {t}")
plt.legend()
plt.show()
