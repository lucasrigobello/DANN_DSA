#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import random as python_random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import loadmat

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

import scipy.stats as stats

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly.express as px

# domain adaptation task
# Ex: Task 0 −→ 1 means working load 0 is the source domain with labeled training samples, and
# working load 1 is the target domain we want to improve model performance on.
# src = 3
# tgt = 0
for src in [3, 2, 1, 0]:
    for tgt in [0, 1, 2, 3]:
        if (src != tgt):
            
            # Dicionario de noem dos arquivos, condição de operação e classe
            orrd = pd.DataFrame([
                    ['100', 3, 0],
                    ['108', 3, 1],
                    ['172', 3, 2],
                    ['212', 3, 3],
                    ['133', 3, 4],
                    ['200', 3, 5],
                    ['237', 3, 6],
                    ['121', 3, 7],
                    ['188', 3, 8],
                    ['225', 3, 9],
                                              
                    ['099', 2, 0],
                    ['107', 2, 1],
                    ['171', 2, 2],
                    ['211', 2, 3],
                    ['132', 2, 4],
                    ['199', 2, 5],
                    ['236', 2, 6],
                    ['120', 2, 7],
                    ['187', 2, 8],
                    ['224', 2, 9],
                    
                    ['098', 1, 0],
                    ['106', 1, 1],
                    ['170', 1, 2],
                    ['210', 1, 3],
                    ['131', 1, 4],
                    ['198', 1, 5],
                    ['235', 1, 6],
                    ['119', 1, 7],
                    ['186', 1, 8],
                    ['223', 1, 9],
                    
                    ['097', 0, 0],
                    ['105', 0, 1],
                    ['169', 0, 2],
                    ['209', 0, 3],
                    ['130', 0, 4],
                    ['197', 0, 5],
                    ['234', 0, 6],
                    ['118', 0, 7],
                    ['185', 0, 8],
                    ['222', 0, 9]], columns = ['name','load','label'])
                                 
            # returns data and labels
            def cwru_data(load, percentage, truncate=120000, length=1024, sample=200, shuffle=False):
                out = []
                data_out = []
                lbls = []
            
                for ii in orrd[orrd['load']==load]['name'].index:
                   
                    lbl = orrd.iloc[ii]['label']
                    file_number = orrd.iloc[ii]['name']
                    file_dirs = './DADOS CWRU/01_Raw/' + file_number + '.mat'
                    data = loadmat(file_dirs)
                    for i in data:
                        # taking only the DE list of values
                        if "X{}_DE_time".format(file_number) == i:

                            # Se for classe de operação normal, realiza reamostragem
                            if file_number in ['097', '098', '099', '100']:
                                print('filtro', file_number)
                                data[i] = data[i][1::4]
                                
                            data = data[i][:truncate] # truncating the length of data
                            data = [data[j:j + length] for j in range(0, len(data) - length + 1 , (len(data) - length)//(sample - 1) )] # sampling
                            #data = np.lib.stride_tricks.sliding_window_view(data.reshape(truncate),1024)[::1024, :]
                                
                            # carrying out fft
                            for k in range(0,len(data)):
                                data_out.append(data[k])
                                fft = abs(np.fft.fft(data[k])[:len(data[k])//2])
                                out.append(fft)
                                lbls.append([lbl])
                        else:
                            pass
            
                # one hot encoding
                #lbls = to_categorical(np.array(lbls))
            
                return data_out, np.array(out), np.array(lbls)
            
            # ============================================================================================
            # preparing data for stage 1 
            data, data_fft, labels = cwru_data(load=src, percentage="full", shuffle=True)
            data_tgt, data_fft_tgt, labels_tgt = cwru_data(load=tgt, percentage="full", shuffle=True)
            
            data_src = np.array(data).reshape(2000,1024)
            data_src = data_src
            label_src = np.array(labels).reshape(2000)
            
            data_tgt = np.array(data_tgt).reshape(2000,1024)
            data_tgt = data_tgt
            label_tgt = np.array(labels_tgt).reshape(2000)
            
            print("Source dimension", data_src.shape, label_src.shape)
            print("Target dimension", data_tgt.shape, label_tgt.shape)
            
            # ============================================================================================
            # preparing data for stage 1 - Transformada de Fourier 
            fft = lambda sig: abs(np.fft.fft(sig)[0:len(sig)//2])/len(sig)
            #fft = lambda sig: abs(np.fft.fft(sig)[0:len(sig)//2])
            
            data_src_fft = np.array([fft(sig) for sig in data_src])
            data_src_fft[1:] = data_src_fft[1:]*2
            
            data_tgt_fft = np.array([fft(sig) for sig in data_tgt])
            data_tgt_fft[1:] = data_tgt_fft[1:]*2
            
            print("Source dimension", data_src_fft.shape, label_src.shape)
            print("Target dimension", data_tgt_fft.shape, label_tgt.shape)
            
            # # ============================================================================================
            # # Expand the last dimension for ease of feeding conv1d
            # data_src = np.expand_dims(data_src, axis=-1)
            # data_src_fft = np.expand_dims(data_src_fft, axis=-1)
            # data_tgt = np.expand_dims(data_tgt, axis=-1)
            # data_tgt_fft = np.expand_dims(data_tgt_fft, axis=-1)
            # print("Source dimension", data_src_fft.shape, data_src.shape)
            # print("Target dimension", data_tgt_fft.shape, data_tgt.shape)
            
            # ============================================================================================
            # ============================================================================================
            # train test Split
            (data_src_fft_train, data_src_fft_test, 
            label_src_train, label_src_test) = train_test_split(data_src_fft, label_src, test_size=0.33, random_state=42)
            print(data_src_fft_train.shape,
                    data_src_fft_test.shape,
                    label_src_train.shape,
                    label_src_test.shape)
            
            (data_tgt_fft_train, data_tgt_fft_test, 
            label_tgt_train, label_tgt_test) = train_test_split(data_tgt_fft, label_tgt, test_size=0.33, random_state=42)
            print(data_tgt_fft_train.shape,
                    data_tgt_fft_test.shape,
                    label_tgt_train.shape,
                    label_tgt_test.shape)
            
            # ============================================================================================
            # ============================================================================================
            # # Definição do modelo e Parametros para Randomized SearchCV
            # # Number of trees in random forest
            # n_estimators = np.linspace(100, 3000, int((3000-100)/200) + 1, dtype=int)
            
            # # Number of features to consider at every split
            # max_features = ['auto', 'sqrt']
            
            # # Maximum number of levels in tree
            # max_depth = [1, 5, 10, 20, 50, 75, 100, 150, 200]
            
            # # Minimum number of samples required to split a node
            # # min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 10, num = 9)]
            # min_samples_split = [1, 2, 5, 10, 15, 20, 30]
            
            # # Minimum number of samples required at each leaf node
            # min_samples_leaf = [1, 2, 3, 4]
            
            # # Method of selecting samples for training each tree
            # bootstrap = [True, False]
            
            # # Criterion
            # criterion=['gini', 'entropy']
            # random_grid = {'n_estimators': n_estimators,
            #                'max_depth': max_depth,
            #                'min_samples_split': min_samples_split,
            #                'min_samples_leaf': min_samples_leaf,
            #                'bootstrap': bootstrap,
            #                'criterion': criterion}
            
            # ============================================================================================
            # Treinamento do modelo
            accs_base_src = []
            accs_base_tgt = []
            for i in range(10):
                python_random.seed(i)
                np.random.seed(i)
                
                # Run training 
                clf = RandomForestClassifier(max_depth=5, random_state=i)
                clf.fit(data_src_fft_train, label_src_train)

                rf_base = RandomForestClassifier()
                # rf_random = RandomizedSearchCV(estimator = rf_base,
                #                                param_distributions = random_grid,
                #                                n_iter = 30, cv = 5,
                #                                verbose=2,
                #                                random_state=42, n_jobs = 4)
                # rf_random.fit(data_src_fft_train, label_src_train)
            
                # Run evaluating source test
                y_pred = clf.predict(data_src_fft_test)
                y_test = label_src_test
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')

                acc_src = accuracy
                
                # Run evaluating source test
                y_pred = clf.predict(data_tgt_fft_test)
                y_test = label_tgt_test
                
                accuracy = accuracy_score(y_test, y_pred)
                # precision = precision_score(y_test, y_pred, average='macro')
                # recall = recall_score(y_test, y_pred, average='macro')

                acc_tgt = accuracy
                print("Accuracy for the baseline model on Source data is", acc_src)
                print("Accuracy for the baseline model on target data is", acc_tgt)
                accs_base_src.append(acc_src)
                accs_base_tgt.append(acc_tgt)
            print("ten run mean", np.mean(accs_base_src))
            print("ten run mean", np.mean(accs_base_tgt))

            np.savetxt("./logs/T{}{}_RF.csv".format(src,tgt), np.array([accs_base_src, accs_base_tgt]), delimiter=";")
            np.savetxt("./logs/T{}{}_mean RF.csv".format(src,tgt), np.array([np.mean(accs_base_src),
                                                   np.mean(accs_base_tgt)]), delimiter=";")
           
            pred = clf.predict(data_src_fft)
            
            cm = confusion_matrix(label_src, pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                         display_labels=np.unique(label_tgt))
            disp.plot()
            plt.xlabel('Classe predita')
            plt.ylabel('Classe verdadeira')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_confusion matrix source validation RandomForest.jpg'.format(src, tgt),
                       dpi = 600)
            plt.show()

            pred = clf.predict(data_tgt_fft)
            
            cm = confusion_matrix(label_tgt, pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                         display_labels=np.unique(label_tgt))
            disp.plot()
            plt.xlabel('Classe predita')
            plt.ylabel('Classe verdadeira')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_confusion matrix target validation RandomForest.jpg'.format(src, tgt),
                       dpi = 600)
            plt.show()
            
            # ============================================================================================
            # Test_t
            dann_values = pd.read_csv('./logs/T{}{}_DANN.csv'.format(src,tgt), sep = ';', header = None).iloc[1,:].values  
            stats.ttest_ind(accs_base_tgt, dann_values)
            np.savetxt("./logs/T{}{}_test_t_RF.csv".format(src,tgt), np.array(
                                                                    stats.ttest_ind(accs_base_tgt, dann_values)
                                                                    ), delimiter=";")

        
