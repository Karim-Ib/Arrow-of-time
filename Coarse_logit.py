import numpy as np
import pandas as pd
from scipy.special import expit
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt


### load data ###


data_txt_B = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\CoarseFeatures.csv', sep=';')
data_B = data_txt_B.to_numpy()

data_txt_J = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\CoarseFeatures.csv', sep=';')
data_J = data_txt_J.to_numpy()

n_data_B = data_B.shape[0]
labels_B = np.zeros(n_data_B)
labels_B[int(n_data_B/2):] = 1

#data_B[int(n_data_B/2):] = np.fliplr(data_B[int(n_data_B/2):])

n_data_J = data_J.shape[0]
labels_J = np.zeros(n_data_J)
labels_J[int(n_data_J/2):] = 1

#data_B[int(n_data_J/2):] = np.fliplr(data_J[int(n_data_J/2):])

### build the classifiers from sk-learn ###
order_B = np.random.random(n_data_B).argsort()
x_train_B = np.take(data_B, order_B, axis=0)
y_train_B = np.take(labels_B, order_B)

order_J = np.random.random(n_data_J).argsort()
x_train_J = np.take(data_J, order_J, axis=0)
y_train_J = np.take(labels_J, order_J)

### http://eointravers.com/post/logistic-overfit/
clf_B = LogisticRegression(solver="liblinear", penalty="l1", C=1)
clf_B.fit(x_train_B, y_train_B)
clf_J = LogisticRegression(solver="liblinear", penalty="l2", C=0.00005)
clf_J.fit(x_train_J, y_train_J)


w_B = clf_B.coef_
w_B = w_B.reshape(-1)
w_mag_B = w_B[:50]
w_nc_B = w_B[50:]
w_J = clf_J.coef_
w_J = w_J.reshape(-1)
w_mag_J = w_J[:50]
w_nc_J = w_J[50:]
time = np.arange(0, 50, 1)

plt.subplot(2, 1, 1)
plt.plot(time, w_mag_B, c="blue", label="MAG")
plt.scatter(time, w_mag_B, c="blue", marker="x")
plt.plot(time, w_nc_B, c="black", label="NNC")
plt.ylabel("weights")
plt.title("B-Protocol")
plt.subplot(2, 1, 2)
plt.plot(time, w_mag_J, c="blue", label="MAG")
plt.plot(time, w_nc_J, c="black", label="NNC")
plt.scatter(time, w_nc_J, c="black", marker="x",  label="NNC")
plt.legend()
plt.xlabel("coarse-grained time")
plt.ylabel("weights")
plt.title("J-Protocol")
plt.show()


print('train accuracy B', clf_B.score(x_train_B, y_train_B))
print('train accuracy J', clf_J.score(x_train_J, y_train_J))

test_txt_B = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\CoarseFeatures.csv', sep=';')
x_test_B = test_txt_B.to_numpy()
y_test_B = np.zeros(x_test_B.shape[0])
y_test_B[int(x_test_B.shape[0] / 2):] = 1
#x_test_B[int(x_test_B.shape[0]/2):] = np.fliplr(x_test_B[int(x_test_B.shape[0]/2):] )
cm_B = confusion_matrix(y_test_B, clf_B.predict(x_test_B))

test_txt_J = pd.read_csv(r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\CoarseFeatures.csv', sep=';')
x_test_J = test_txt_J.to_numpy()
y_test_J = np.zeros(x_test_J.shape[0])
#x_test_J[int(x_test_J.shape[0]/2):] = np.fliplr(x_test_J[int(x_test_J.shape[0]/2):] )
cm_J = confusion_matrix(y_test_J, clf_J.predict(x_test_J))



#print("report for J protocol", classification_report(y_test_J, clf_J.predict(x_test_J)))
#print("report for B protocol", classification_report(y_test_B, clf_B.predict(x_test_B)))
