import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#import TF models
gate = tf.keras.models.load_model(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\gate')
model_J = tf.keras.models.load_model(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\binary_clf')

#train logistic model TODO:learn how to set weights on sklearn logistic regression to skip this

data_txt = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\Data.csv', sep=';')
labels_txt = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\Data_labels.csv', sep=';')
data = data_txt.to_numpy()
labels = labels_txt.to_numpy()

n_data = data.shape[0]
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)

model_B = LogisticRegression(solver="liblinear", penalty="l1",C=9.9 * 10 ** -3)
model_B.fit(x_train, y_train)


### load testing data
data_B_test = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\Data.csv', sep=';')
data_B_test = data_B_test.to_numpy()
labels_B_test = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\Data_labels.csv', sep=';')
labels_B_test = labels_B_test.to_numpy()

data_J_test = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\Data.csv', sep=';')
data_J_test = data_J_test.to_numpy()
labels_J_test = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\Data_labels.csv', sep=';')
labels_J_test = labels_J_test.to_numpy()



## STEP 1: gating

gate_B = (gate.predict(np.array(data_B_test).reshape(-1, 500, 10, 1))).reshape(-1)
gate_J = (gate.predict(np.array(data_J_test).reshape(-1, 500, 10, 1))).reshape(-1)


correct_B = np.argwhere(gate_B < 0.5)
correct_J = np.argwhere(gate_J > 0.5)

print("correct gated B", len(correct_B) / len(labels_B_test))
print("correct gated J", len(correct_J) / len(labels_J_test))

## STEP 2: ask experts

B_logit = model_B.predict(data_B_test)
correct_B_logit = np.argwhere(B_logit == labels_B_test.reshape(-1))
print("correct  B Logit Predictions", len(correct_B_logit) / len(labels_B_test))

B_cnn = model_J.predict(np.array(data_B_test).reshape(-1, 500, 10, 1))
B_cnn[B_cnn < 0.5] = 0
B_cnn[B_cnn >= 0.5] = 1
correct_B_cnn = np.argwhere(B_cnn == labels_B_test)
print("correct B cnn Predictions", len(correct_B_cnn) / len(labels_B_test))


J_logit = model_B.predict(data_J_test)
correct_J_logit = np.argwhere(J_logit == labels_J_test.reshape(-1))
print("correct  J Logit Predictions", len(correct_J_logit) / len(labels_J_test))

J_cnn = model_J.predict(np.array(data_J_test).reshape(-1, 500, 10, 1))
J_cnn[J_cnn < 0.5] = 0
J_cnn[J_cnn >= 0.5] = 1
correct_J_cnn = np.argwhere(J_cnn == labels_J_test)
print("correct J cnn Predictions", len(correct_J_cnn) / len(labels_J_test))

## STEP 3: mix experts
results_B = (1 - gate_B).T * B_logit + gate_B * B_cnn.T
results_J =  gate_J * J_cnn.T + (1 - gate_J) * J_logit

## STEP 4: evaluate results

results_B[results_B < 0.5] = 0
results_B[results_B >= 0.5] = 1
correct_B_moe = np.argwhere(results_B == labels_B_test.reshape(-1))

results_J[results_J < 0.5] = 0
results_B[results_B >= 0.5] = 1
correct_J_moe = np.argwhere(results_J == labels_J_test.reshape(-1))

print("MoE correct B Predictions", len(correct_B_moe) / len(labels_B_test))
print("MoE correct J Predictions", len(correct_J_moe) / len(labels_J_test))


