import numpy as np
import os, glob
from Ising_1D import get_work, set_cos, get_work_sum, Hamiltonian_Ising


def get_data_work(set = 'train', Protocol = 1):
    ###
    # 0 = J, 1 = B
    N_spins = 10 ** 1
    N_steps = N_spins * 50
    B = 20
    B_list = set_cos(B, N_steps)

    ### initialize as python array -> convert to numpy after should be O(n) - not optimal but should work for now - takes about 60-90s
    data = []
    data_labels = []
    work_fw = []
    work_bw = []

    # TODO: change to relativ path at some point https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python

    if Protocol == 1:
        if set == 'train':
            ## B-Paths train
            path_backward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\backward\\'
            path_forward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\forward\\'
        if set == 'test':
            ## B-Paths test
            path_backward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\backward\\'
            path_forward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\forward\\'


    if Protocol == 0:
        if set == 'train':
            ### J-Paths
            path_backward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\backward\\'
            path_forward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\forward\\'
        if set == 'test':
            ### J-Paths
            path_backward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\backward\\'
            path_forward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\forward\\'

    for filename in glob.glob(os.path.join(path_backward, '*.csv')):         #iterates from 0.02.._0 to 0.1_999
        x = np.genfromtxt(filename, delimiter=';')
        data.append(x.flatten())
        data_labels.append(1)
        work_bw.append(-get_work(np.flipud(B_list), np.flipud(x), Protocol))
        #work_bw.append(-get_work_sum(Hamiltonian_Ising, np.flipud(B_list), J, np.flipud(x), 1))

    for filename in glob.glob(os.path.join(path_forward, '*.csv')):
        x = np.genfromtxt(filename, delimiter=';')
        data.append(x.flatten())
        data_labels.append(0)
        work_fw.append(get_work(B_list, x, Protocol))
        #work_fw.append(get_work_sum(Hamiltonian_Ising, B_list, J, x, 1))


    data_labels = np.array(data_labels)

    if Protocol == 1:
        if set == 'train':
            np.savetxt(f'Data_B/Data.csv', data,  delimiter=";")
            np.savetxt(f'Data_B/Data_labels.csv', data_labels,  delimiter=";")
            np.savetxt(f'Data_B/work_fw.csv', work_fw, delimiter=";")
            np.savetxt(f'Data_B/work_bw.csv', work_bw, delimiter=";")
        if set == 'test':
            np.savetxt(f'Data_B_test/Data.csv', data,  delimiter=";")
            np.savetxt(f'Data_B_test/Data_labels.csv', data_labels,  delimiter=";")
            np.savetxt(f'Data_B_test/work_fw.csv', work_fw, delimiter=";")
            np.savetxt(f'Data_B_test/work_bw.csv', work_bw, delimiter=";")



    if Protocol == 0:
        if set == 'train':
            np.savetxt(f'Data_J/Data.csv', data,  delimiter=";")
            np.savetxt(f'Data_J/Data_labels.csv', data_labels,  delimiter=";")
            np.savetxt(f'Data_J/work_fw.csv', work_fw, delimiter=";")
            np.savetxt(f'Data_J/work_bw.csv', work_bw, delimiter=";")
        if set == 'test':
            np.savetxt(f'Data_J_test/Data.csv', data,  delimiter=";")
            np.savetxt(f'Data_J_test/Data_labels.csv', data_labels,  delimiter=";")
            np.savetxt(f'Data_J_test/work_fw.csv', work_fw, delimiter=";")
            np.savetxt(f'Data_J_test/work_bw.csv', work_bw, delimiter=";")

def get_temp_split(set = 'train', Protocol = 1):

    if Protocol == 1:
        if set == 'train':
            path_backward_cold = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\backward\B_b_0.1*'
            path_forward_cold = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\forward\B_f_0.1*'

            path_backward_med = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\backward\B_b_0.03*'
            path_forward_med = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\forward\B_f_0.03*'

            path_backward_hot = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\backward\B_b_0.02*'
            path_forward_hot = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B\forward\B_f_0.02*'
        if set == 'test':
            path_backward_cold = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\backward\B_b_0.1*'
            path_forward_cold = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\forward\B_f_0.1*'

            path_backward_med = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\backward\B_b_0.03*'
            path_forward_med = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\forward\B_f_0.03*'

            path_backward_hot = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\backward\B_b_0.02*'
            path_forward_hot = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_test\forward\B_f_0.02*'


        data_cold = []
        data_med = []
        data_hot = []

        labels_cold = []
        labels_med = []
        labels_hot = []

        for filename in glob.glob(path_forward_cold):
            x = np.genfromtxt(filename, delimiter=';')
            data_cold.append(x.flatten())
            labels_cold.append(0)

        for filename in glob.glob(path_backward_cold):

            x = np.genfromtxt(filename, delimiter=';')
            data_cold.append(x.flatten())
            labels_cold.append(1)



        if set == 'train':
            np.savetxt(f'Data_B/Data_cold.csv', data_cold, delimiter=";")
            np.savetxt(f'Data_B/labels_cold.csv', labels_cold, delimiter=";")
        if set == 'test':
            np.savetxt(f'Data_B_test/Data_cold.csv', data_cold, delimiter=";")
            np.savetxt(f'Data_B_test/labels_cold.csv', labels_cold, delimiter=";")


        del data_cold, labels_cold


        for filename in glob.glob(path_forward_med):
            x = np.genfromtxt(filename, delimiter=';')
            data_med.append(x.flatten())
            labels_med.append(0)

        for filename in glob.glob(path_backward_med):
            x = np.genfromtxt(filename, delimiter=';')
            data_med.append(x.flatten())
            labels_med.append(1)



        if set == 'train':
            np.savetxt(f'Data_B/Data_med.csv', data_med, delimiter=";")
            np.savetxt(f'Data_B/labels_med.csv', labels_med, delimiter=";")
        if set == 'test':
            np.savetxt(f'Data_B_test/Data_med.csv', data_med, delimiter=";")
            np.savetxt(f'Data_B_test/labels_med.csv', labels_med, delimiter=";")

        del data_med, labels_med

        for filename in glob.glob(path_forward_hot):
            x = np.genfromtxt(filename, delimiter=';')
            data_hot.append(x.flatten())
            labels_hot.append(0)

        for filename in glob.glob(path_backward_hot):
            x = np.genfromtxt(filename, delimiter=';')
            data_hot.append(x.flatten())
            labels_hot.append(1)


        if set == 'train':
            np.savetxt(f'Data_B/Data_hot.csv', data_hot, delimiter=";")
            np.savetxt(f'Data_B/labels_hot.csv', labels_hot, delimiter=";")
        if set == 'test':
            np.savetxt(f'Data_B_test/Data_hot.csv', data_hot, delimiter=";")
            np.savetxt(f'Data_B_test/labels_hot.csv', labels_hot, delimiter=";")

        del data_hot, labels_hot

    if Protocol == 0:
        if set == 'train':
            path_backward_cold = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\backward\J_b_0.1*'
            path_forward_cold = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\forward\J_f_0.1*'

            path_backward_med = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\backward\J_b_0.03*'
            path_forward_med = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\forward\J_f_0.03*'

            path_backward_hot = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\backward\J_b_0.02*'
            path_forward_hot = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J\forward\J_f_0.02*'
        if set == 'test':
            path_backward_cold = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\backward\J_b_0.1*'
            path_forward_cold = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\forward\J_f_0.1*'

            path_backward_med = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\backward\J_b_0.03*'
            path_forward_med = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\forward\J_f_0.03*'

            path_backward_hot = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\backward\J_b_0.02*'
            path_forward_hot = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_test\forward\J_f_0.02*'

        data_cold = []
        data_med = []
        data_hot = []

        labels_cold = []
        labels_med = []
        labels_hot = []

        for filename in glob.glob(path_forward_cold):
            x = np.genfromtxt(filename, delimiter=';')
            data_cold.append(x.flatten())
            labels_cold.append(0)
        for filename in glob.glob(path_backward_cold):
            x = np.genfromtxt(filename, delimiter=';')
            data_cold.append(x.flatten())
            labels_cold.append(1)


        if set == 'train':
            np.savetxt(f'Data_J/Data_cold.csv', data_cold, delimiter=";")
            np.savetxt(f'Data_J/labels_cold.csv', labels_cold, delimiter=";")
        if set == 'test':
            np.savetxt(f'Data_J_test/Data_cold.csv', data_cold, delimiter=";")
            np.savetxt(f'Data_J_test/labels_cold.csv', labels_cold, delimiter=";")
        del data_cold, labels_cold

        for filename in glob.glob(path_forward_med):
            x = np.genfromtxt(filename, delimiter=';')
            data_med.append(x.flatten())
            labels_med.append(0)

        for filename in glob.glob(path_backward_med):
            x = np.genfromtxt(filename, delimiter=';')
            data_med.append(x.flatten())
            labels_med.append(1)

        if set == 'train':
            np.savetxt(f'Data_J/Data_med.csv', data_med, delimiter=";")
            np.savetxt(f'Data_J/labels_med.csv', labels_med, delimiter=";")
        if set == 'test':
            np.savetxt(f'Data_J_test/Data_med.csv', data_med, delimiter=";")
            np.savetxt(f'Data_J_test/labels_med.csv', labels_med, delimiter=";")

        del data_med, labels_med

        for filename in glob.glob(path_forward_hot):
            x = np.genfromtxt(filename, delimiter=';')
            data_hot.append(x.flatten())
            labels_hot.append(0)
        for filename in glob.glob(path_backward_hot):
            x = np.genfromtxt(filename, delimiter=';')
            data_hot.append(x.flatten())
            labels_hot.append(1)


        if set == 'train':
            np.savetxt(f'Data_J/Data_hot.csv', data_hot, delimiter=";")
            np.savetxt(f'Data_J/labels_hot.csv', labels_hot, delimiter=";")
        if set == 'test':
            np.savetxt(f'Data_J_test/Data_hot.csv', data_hot, delimiter=";")
            np.savetxt(f'Data_J_test/labels_hot.csv', labels_hot, delimiter=";")

        del data_hot, labels_hot

def GetCoarse(input_path, fileName):
    input = np.genfromtxt(input_path+fileName, delimiter=";")
    Ncoarse = 10
    n_data = input.shape[0]
    input[:int(n_data/2), :] = np.fliplr(input[:int(n_data/2), :])
    mag_list = np.empty((input.shape[0], 500))
    NC_list = np.empty((input.shape[0], 500))
    coarse_list_mag = np.empty((input.shape[0], 50))
    coarse_list_NC = np.empty((input.shape[0], 50))

    for rows in range(input.shape[0]):
        ### Coarse Graninig magnetization
        temp = input[rows, :].reshape(500, 10)
        mag_list[rows, :] = temp.sum(axis=1)
        temp = mag_list[rows, :].reshape(-1, Ncoarse)
        coarse_list_mag[rows, :] = temp.sum(axis=1)

        ### Coarse graninig nearest neighbour correlations
        temp = input[rows, :].reshape(500, 10)
        NC_list[rows, :] = (temp * np.roll(temp, 1, axis=1)).sum(axis=1)
        temp = NC_list[rows, :].reshape(-1, Ncoarse)
        coarse_list_NC[rows, :] = temp.sum(axis=1)

    features = np.hstack((coarse_list_mag, coarse_list_NC))
    np.savetxt(input_path+"CoarseFeatures.csv", features, delimiter=";")
    return

def GetCoarse_J(input_path, fileName):
    input = np.genfromtxt(input_path+fileName, delimiter=";")
    n_data = input.shape[0]

    bw = input[(2 * int(n_data / 6)):(3 * int(n_data / 6)), :]
    bw = np.fliplr(bw)
    fw = input[(5 * int(n_data / 6)):, :]
    input = np.vstack((bw, fw))

    Ncoarse = 10
    n_data = input.shape[0]
    mag_list = np.empty((input.shape[0], 500))
    NC_list = np.empty((input.shape[0], 500))
    coarse_list_mag = np.empty((input.shape[0], 50))
    coarse_list_NC = np.empty((input.shape[0], 50))

    for rows in range(input.shape[0]):
        ### Coarse Graninig magnetization
        temp = input[rows, :].reshape(500, 10)
        mag_list[rows, :] = temp.sum(axis=1)
        temp = mag_list[rows, :].reshape(-1, Ncoarse)
        coarse_list_mag[rows, :] = temp.sum(axis=1)

        ### Coarse graninig nearest neighbour correlations
        temp = input[rows, :].reshape(500, 10)
        NC_list[rows, :] = (temp * np.roll(temp, 1, axis=1)).sum(axis=1)
        temp = NC_list[rows, :].reshape(-1, Ncoarse)
        coarse_list_NC[rows, :] = temp.sum(axis=1)
    features = np.hstack((coarse_list_mag, coarse_list_NC))
    np.savetxt(input_path+"CoarseFeatures.csv", features, delimiter=";")
    return
def get_data_work_full(Protocol = 1):
    ###
    # 0 = J, 1 = B
    N_spins = 10 ** 1
    N_steps = N_spins * 50
    B = 20
    B_list = set_cos(B, N_steps)

    ### initialize as python array -> convert to numpy after should be O(n) - not optimal but should work for now - takes about 60-90s
    data = []
    data_labels = []
    work_fw = []
    work_bw = []

    # TODO: change to relativ path at some point https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python

    if Protocol == 1:

        ## B-Paths train
        path_backward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_full\backward\\'
        path_forward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_B_full\forward\\'



    if Protocol == 0:

        ### J-Paths
        path_backward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_full\backward\\'
        path_forward = r'C:\Users\Nutzer\Documents\SS2021\Projekt Statistische Physik\Ising_Model\Data_J_full\forward\\'


    for filename in glob.glob(os.path.join(path_backward, '*.csv')):         #iterates from 0.02.._0 to 0.1_999
        x = np.genfromtxt(filename, delimiter=';')
        data.append(x.flatten())
        data_labels.append(1)
        work_bw.append(get_work(np.flipud(B_list), np.flipud(x), Protocol))
        #work_bw.append(-get_work_sum(Hamiltonian_Ising, np.flipud(B_list), J, np.flipud(x), 1))

    for filename in glob.glob(os.path.join(path_forward, '*.csv')):
        x = np.genfromtxt(filename, delimiter=';')
        data.append(x.flatten())
        data_labels.append(0)
        work_fw.append(-get_work(B_list, x, Protocol))
        #work_fw.append(get_work_sum(Hamiltonian_Ising, B_list, J, x, 1))


    data_labels = np.array(data_labels)

    if Protocol == 1:

        np.savetxt(f'Data_B_full/Data.csv', data,  delimiter=";")
        np.savetxt(f'Data_B_full/Data_labels.csv', data_labels,  delimiter=";")
        np.savetxt(f'Data_B_full/work_fw.csv', work_fw, delimiter=";")
        np.savetxt(f'Data_B_full/work_bw.csv', work_bw, delimiter=";")




    if Protocol == 0:

        np.savetxt(f'Data_J_full/Data.csv', data,  delimiter=";")
        np.savetxt(f'Data_J_full/Data_labels.csv', data_labels,  delimiter=";")
        np.savetxt(f'Data_J_full/work_fw.csv', work_fw, delimiter=";")
        np.savetxt(f'Data_J_full/work_bw.csv', work_bw, delimiter=";")





'''get_temp_split('train', 0)
get_temp_split('train', 1)
get_data_work('train', 0)
get_data_work('train', 1)
'''

'''get_temp_split('test', 0)
get_temp_split('test', 1)
get_data_work('test', 0)
get_data_work('test', 1)'''

path_b = f'Data_B_full/'
path_J = f'Data_J_full/'
path_b_test = f'Data_B_test/'
path_J_test = f'Data_J_test/'
file = f'Data.csv'

#GetCoarse(path_b, file)
#GetCoarse(path_b_test, file)
GetCoarse_J(path_b, file)
#GetCoarse(path_J_test, file)

#get_data_work_full(0)
#get_data_work_full(1)