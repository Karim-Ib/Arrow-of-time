import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


data_B = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\Data.csv', sep=';')
data_J = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\Data.csv', sep=';')
data_B = data_B.to_numpy()
data_J = data_J.to_numpy()
data = np.concatenate((data_B, data_J), axis=0)
print(data.shape)
labels = np.zeros(data.shape[0])
labels[int(data.shape[0] / 2) :] = 1
print(np.argwhere(np.isnan(data)))
data = np.array(data).reshape(-1, 500, 10, 1)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4)


input = tf.keras.layers.Input((500, 10, 1))
con1 = tf.keras.layers.Conv2D(4, kernel_size=(2,2), padding="SAME", activation="relu")(input)
dropout = tf.keras.layers.Dropout(0.1)(con1)
mp1 = tf.keras.layers.MaxPooling2D((2, 2))(dropout)
flatten = tf.keras.layers.Flatten()(mp1)
dense1 = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1))(flatten)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense1)

gate = tf.keras.Model(inputs=input, outputs=output)
gate.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
gate.summary()
history = gate.fit(x_train, y_train,  epochs=7, verbose=1, validation_split=0.2)
score = gate.evaluate(x_test, y_test, verbose=1)
gate.save(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\gate')
p = gate.predict(x_test)

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


#plt.plot(np.linspace(0, len(y_test), len(y_test)), p)
plt.show()