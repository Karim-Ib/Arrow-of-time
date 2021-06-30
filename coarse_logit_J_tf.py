import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import  to_categorical
import matplotlib.pyplot as plt
import pandas as pd
from Ising_1D import calc_sigmoid, get_work, set_cos

#load train data

data_txt = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\CoarseFeatures.csv', sep=';')

x = data_txt.to_numpy()
n_data = x.shape[0]
n_features = x.shape[1]
y = np.zeros(n_data)
y[:int(n_data/2)] = 1
print(n_data, n_features)


#reverse backwards trajectories already done in prep
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


#define model
clf = Sequential()
clf.add(Dense(1, activation="sigmoid", kernel_regularizer=l2(2*10**-5), input_dim=(n_features)))

#compile model
clf.summary()
optimizer_lr = tf.keras.optimizers.Adam(learning_rate=0.001)
clf.compile(optimizer=optimizer_lr, loss="binary_crossentropy", metrics=["accuracy"])

#fit model
history = clf.fit(x, y, epochs=50, verbose=1, validation_split=0.2)

#get weights

weights = clf.layers[0].get_weights()[0]


#visualize
clf.save(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\coarse_logit_J_TF')

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

#weights plot

plt.scatter(np.arange(0, int(len(weights)/2), 1), weights[:int(len(weights)/2)], c="blue")
plt.scatter(np.arange(0, int(len(weights)/2), 1), weights[int(len(weights)/2):], c="black")
plt.show()