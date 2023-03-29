#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import random as python_random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import loadmat

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Conv1D, Flatten, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K

import scipy.stats as stats

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly.express as px

# Confirmação de GPU ativa e reconhecida
tf.config.experimental.list_physical_devices()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# domain adaptation task
# Ex: Task 0 −→ 1 means working load 0 is the source domain with labeled training samples, and
# working load 1 is the target domain we want to improve model performance on.
# src = 3
# tgt = 0
for src in [1]:
    for tgt in [2]:
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
            
            # ============================================================================================
            # Expand the last dimension for ease of feeding conv1d
            data_src = np.expand_dims(data_src, axis=-1)
            data_src_fft = np.expand_dims(data_src_fft, axis=-1)
            data_tgt = np.expand_dims(data_tgt, axis=-1)
            data_tgt_fft = np.expand_dims(data_tgt_fft, axis=-1)
            print("Source dimension", data_src_fft.shape, data_src.shape)
            print("Target dimension", data_tgt_fft.shape, data_tgt.shape)
            
            # ============================================================================================
            # Preparação para plotagem na linha do tempo e frequencia
            fs = 12000
            N = 1024
            tend = N/fs
            dt = tend/N
            print('df = {}'.format(1/tend))
            t = np.array(range(0,N))*dt
            print(t)
            
            f = np.array(range(0,N//2))*fs/N
            
            # Plotagem da aceleração na linha do tempo
            plt.figure(figsize=(8, 4))
            for i in range(10):
                plt.plot(t, data_tgt[200*i], label="classe: " + str(i))
                assert(label_tgt[200*i] == i)
            plt.legend(loc = 'upper right',
                      bbox_to_anchor=(1.3, 1.02))
            plt.xlabel('Tempo (s)')
            plt.ylabel("Amplitude ($m/s^2$)")
            plt.tight_layout()
            plt.savefig('./imagens/C{}_todos as classes amosta no tempo.jpg'.format(tgt), dpi = 600)
            
            # Plotagem dos coeficientes de Fourier no domínio da frequência
            plt.figure(figsize=(8, 4))
            for i in range(10):
                plt.plot(f, data_tgt_fft[200*i], label="class: " + str(i))
                assert(label_tgt[200*i] == i)
            plt.legend(loc = 'upper right',
                      bbox_to_anchor=(1.25, 1.02))
            plt.xlabel('Frequência (Hz)')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.savefig('./imagens/C{}_todos as classes amosta frequencia.jpg'.format(tgt), dpi = 600)
            
            # ============================================================================================
            # PCA Plotting 
            pca = PCA(n_components=7)
            pca_model = pca.fit(data_src_fft.reshape(2000,512))
            Xfactor_train_source = pca_model.transform(data_src_fft.reshape(2000,512))
            Xfactor_train_target = pca_model.transform(data_tgt_fft.reshape(2000,512))
            
            Xfactor_source = pd.DataFrame(Xfactor_train_source).reset_index(drop=True)
            Xfactor_target = pd.DataFrame(Xfactor_train_target).reset_index(drop=True)
            y_train_source = pd.DataFrame(label_src).reset_index(drop=True)
            y_train_source['Condition']='Source'
            Xfactor_source[['Classe', 'Condition']]= y_train_source
            y_train_target = pd.DataFrame(label_tgt).reset_index(drop=True)
            y_train_target['Condition']='Target'
            Xfactor_target[['Classe', 'Condition']]= y_train_target
            
            Xfactor_train_global = pd.concat([Xfactor_source, Xfactor_target])
            Xfactor_train_global.reset_index(inplace=True, drop=True)
            Xfactor_train_global
            
            print(
                "explained variance ratio (first two components): %s"
                % str(pca_model.explained_variance_ratio_)
            )
            
            plt.figure(figsize=(8, 4))
            plt.subplot(1,2,1)
            ax = sns.scatterplot(data=Xfactor_train_global[Xfactor_train_global['Condition']=='Source'],
                            x = 0, y = 1, hue = 'Classe', 
                            palette = "tab10")
            ax.legend_.remove()
            plt.xlabel('comp-1')
            plt.ylabel('comp-2')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_PCA_source_input.jpg'.format(src, tgt),
                       dpi = 600)
            
            plt.figure(figsize=(8, 4))
            plt.subplot(1,2,1)
            ax = sns.scatterplot(data=Xfactor_train_global[Xfactor_train_global['Condition']=='Target'],
                            x = 0, y = 1, hue = 'Classe', 
                            palette = "tab10")
            ax.legend_.remove()
            plt.xlabel('comp-1')
            plt.ylabel('comp-2')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_PCA_target_input.jpg'.format(src, tgt),
                       dpi = 600)
            
            plt.figure(figsize=(8, 4))
            plt.subplot(1,2,1)
            ax = sns.scatterplot(data=Xfactor_train_global,
                            x = 0, y = 1, hue = 'Condition', 
                            palette = "tab10")
            ax.legend_.remove()
            plt.xlabel('comp-1')
            plt.ylabel('comp-2')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_PCA_conditions_distribution_input.jpg'.format(src, tgt),
                       dpi = 600)
            
            fig = px.scatter_3d(Xfactor_train_global, x=0, y=1, z=2,
                          color='Condition')
            fig.show()

            # ============================================================================================
            # TSNE 
            aux = np.append(data_src_fft.reshape(2000,512), data_tgt_fft.reshape(2000,512), axis = 0)
            aux.shape

            tsne = TSNE(n_components=2, verbose=1, random_state=123)
            z = tsne.fit_transform(aux)

            yaux = np.append(label_src, label_tgt)

            df = pd.DataFrame()
            df["y"] = yaux
            df["y2"] = 'Source'
            df.loc[2000:, 'y2'] = 'Target'
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]

            plt.figure(figsize=(8, 4))
            plt.subplot(1,2,1)
            ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                            palette=sns.color_palette("hls", 10),
                            data=df)
            ax.legend_.remove()
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_tSNE_distribution_classes_input.jpg'.format(src, tgt),
                       dpi = 600)

            plt.figure(figsize=(8, 4))
            plt.subplot(1,2,1)
            ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y2.tolist(),
                            palette=sns.color_palette("hls", 2),
                            data=df)
            ax.legend_.remove()
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_tSNE_distribution_conditions_input.jpg'.format(src, tgt),
                       dpi = 600)
        
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
            # Definição do modelo Base Line
            def feature_extractor(x):
                h = Conv1D(10, 3, padding='same', activation="relu")(x)
                h = Dropout(0.5)(h)
                h = Conv1D(10, 3, padding='same', activation="relu")(h)
                h = Dropout(0.5)(h)
                h = Conv1D(10, 3, padding='same', activation="relu")(h)
                h = Dropout(0.5)(h)
                h = Flatten()(h)
                h = Dense(256, activation='relu', name = 'feature')(h)
                return h

            def clf(x):    
                h = Dense(256, activation='relu')(x)
                h = Dense(10, activation='softmax', name="clf")(h)
                return h
            
            def baseline():
                input_dim = 512    
                inputs = Input(shape=(input_dim, 1))
                features = feature_extractor(inputs)
                logits = clf(features)
                baseline_model = Model(inputs=inputs, outputs=logits)
                adam = Adam(learning_rate=0.0001)
                baseline_model.compile(optimizer=adam,
                          loss=['sparse_categorical_crossentropy'], metrics=['accuracy'])
                return baseline_model
            
            # Set seed
            accs_base_src = []
            accs_base_tgt = []
            #accs_base = []

            for i in range(10):
                python_random.seed(i)
                np.random.seed(i)
                tf.random.set_seed(i)
                baseline_model = baseline()
                # Run training 
                history_base = baseline_model.fit(data_src_fft_train, label_src_train,
                                                  batch_size=128,
                                                  epochs=1000,
                                                  shuffle=True,
                                                  verbose=False)
                # Run evaluating
                score, acc_src = baseline_model.evaluate(data_src_fft_test, label_src_test, batch_size=128)
                score, acc_tgt = baseline_model.evaluate(data_tgt_fft_test, label_tgt_test, batch_size=128)
                print("Accuracy for the baseline model on Source data is", acc_src)
                print("Accuracy for the baseline model on target data is", acc_tgt)
                accs_base_src.append(acc_src)
                accs_base_tgt.append(acc_tgt)
            print("ten run mean", np.mean(accs_base_src))
            print("ten run mean", np.mean(accs_base_tgt))

            np.savetxt("./logs/T{}{}_baseline.csv".format(src,tgt), np.array([accs_base_src, accs_base_tgt]), delimiter=";")
            np.savetxt("./logs/T{}{}_mean_baseline.csv".format(src,tgt), np.array([np.mean(accs_base_src),
                                                   np.mean(accs_base_tgt)]), delimiter=";")

            plt.figure(figsize=(8, 4))
            plt.plot(history_base.epoch, history_base.history["loss"], 'orange', label='Training loss')
            plt.title('Training loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            
            plt.figure(figsize=(8, 4))
            plt.plot(history_base.epoch, history_base.history["accuracy"], 'g', label='Training accuracy')
            plt.title('Training accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('accuracy')
            plt.legend()
            plt.show()
            
            pred = baseline_model.predict(data_src_fft)
            pred = [np.argmax(x) for x in pred]
            
            cm = confusion_matrix(label_src, pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                         display_labels=np.unique(label_tgt))
            disp.plot()
            plt.xlabel('Classe predita')
            plt.ylabel('Classe verdadeira')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_confusion matrix source validation Baseline.jpg'.format(src, tgt),
                       dpi = 600)
            plt.show()

            pred = baseline_model.predict(data_tgt_fft)
            pred = [np.argmax(x) for x in pred]
            
            cm = confusion_matrix(label_tgt, pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                         display_labels=np.unique(label_tgt))
            disp.plot()
            plt.xlabel('Classe predita')
            plt.ylabel('Classe verdadeira')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_confusion matrix target validation Baseline.jpg'.format(src, tgt),
                       dpi = 600)
            plt.show()
            
            
            # ============================================================================================
            # ============================================================================================
            # # Gradient Reverse Layer(GRL) Layer

            @tf.custom_gradient
            def grad_reverse(x):
                y = tf.identity(x)
                def custom_grad(dy):
                    return -dy
                return y, custom_grad
            
            class GradReverse(tf.keras.layers.Layer):
                def __init__(self):
                    super().__init__()
            
                def call(self, x):
                    return grad_reverse(x)
 
            def discriminator(x):    
                h = Dense(1024, activation='relu')(x)
                h = Dense(1024, activation='relu')(h)
                h = Dense(2, activation='softmax', name="dis")(h)
                return h

            def grl():
                #GRL strategy
                #returns: the classification branch, the discriminator branch
                
                input_dim = 512
                
                ### Define inputs
                inputs = Input(shape=(input_dim, 1))
                ### Get features
                features = feature_extractor(inputs)
                ### Get classification logits
                logits = clf(features)
                ### Define the classification branch model
                clf_branch = Model(inputs=inputs, outputs=logits)
                adam = Adam(learning_rate=0.0001)   
                clf_branch.compile(optimizer=adam,
                          loss={'clf': 'sparse_categorical_crossentropy'}, metrics=['accuracy'])
                
                ### Define the classification branch model    
                features_rev = GradReverse()(features)
                logits_da = discriminator(features_rev)
            
                da_branch = Model(inputs=inputs, outputs=logits_da)
                adam_da = Adam(learning_rate=0.0001)   
                da_branch.compile(optimizer=adam_da,
                          loss={'dis': 'sparse_categorical_crossentropy'}, metrics=['accuracy'])
                return clf_branch, da_branch

            ### Some constants
            NUM_EPOCH = 1000
            BATCH_SIZE = 128
            DATASET_SIZE = 1340
            
            accs_src = []
            accs_tgt = []
            history_tr = pd.DataFrame([], columns = ['loss', 'accuracy'])
            history_val = pd.DataFrame([], columns = ['loss', 'accuracy'])
            
            import random as python_random
            
            for i in range(10):
                
                python_random.seed(i)
                np.random.seed(i)
                tf.random.set_seed(i)
                clf_branch, da_branch = grl()
                
                ### Iterate over 
                #for j in range(NUM_EPOCH * (DATASET_SIZE // BATCH_SIZE)):
                for j in range(NUM_EPOCH):
                    
                    ### Randomly fetch training data
                    idx_src = np.random.choice(DATASET_SIZE, size=BATCH_SIZE, replace=False)
                    idx_tgt = np.random.choice(DATASET_SIZE, size=BATCH_SIZE, replace=False)
                    batch_src, batch_y = data_src_fft_train[idx_src], label_src_train[idx_src]
                    
                    ### We don't use any label from target domain
                    batch_tgt = data_tgt_fft_train[idx_tgt] 
                    
                    ########## the training code for clf_branch ###################
                    result = clf_branch.train_on_batch(batch_src, batch_y)
                    result = clf_branch.evaluate(data_src_fft_test, label_src_test, batch_size=128, verbose = False)
                    history_tr = pd.concat([history_tr, pd.DataFrame([result], columns = ['loss', 'accuracy'])])
                    
                    result = clf_branch.evaluate(data_tgt_fft_test, label_tgt_test, batch_size=128, verbose = False)
                    history_val = pd.concat([history_val, pd.DataFrame([result], columns = ['loss', 'accuracy'])])
                    
                    ########## the training code for discriminator branch #########
                    dis_y = np.concatenate([np.zeros_like(batch_y), np.ones_like(batch_y)], axis=0)
                    da_branch.train_on_batch(np.concatenate([batch_src, batch_tgt], axis=0), dis_y)
            
                ### Final results
                score, acc_src = clf_branch.evaluate(data_src_fft_test, label_src_test, batch_size=128)
                score, acc_tgt = clf_branch.evaluate(data_tgt_fft_test, label_tgt_test, batch_size=128)
                print("Final Accuracy src", acc_src)
                print("Final Accuracy tgt", acc_tgt)
                accs_src.append(acc_src)
                accs_tgt.append(acc_tgt)
                
            history_tr.reset_index(inplace = True, drop = True)
            history_val.reset_index(inplace = True, drop = True)
            print("ten run mean src", np.mean(accs_src))
            print("ten run mean tgt", np.mean(accs_tgt))
            
            np.savetxt("./logs/T{}{}_DANN.csv".format(src,tgt), np.array([accs_src, accs_tgt]), delimiter=";")
            np.savetxt("./logs/T{}{}_mean_DANN.csv".format(src,tgt), np.array([np.mean(accs_src),
                                                                                np.mean(accs_tgt)]), delimiter=";")

            stats.ttest_ind(accs_base_tgt, accs_tgt)
            np.savetxt("./logs/T{}{}_test_t.csv".format(src,tgt), np.array(
                                                                    stats.ttest_ind(accs_base_tgt, accs_tgt)
                                                                    ), delimiter=";")

            # ============================================================================================
            # Plot Loss e Accuracy nas Epochs
            plt.figure(figsize=(8, 4))
            #plt.subplot(2,1,1)
            plt.plot(history_tr.index[0:1000], history_tr["loss"][0:1000], 'orange', label='Perda de teste "source"')
            plt.plot(history_val.index[0:1000], history_val["loss"][0:1000], 'b', label='Perda de teste "target"')
            #plt.title('Training loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_epochs_loss.jpg'.format(src, tgt),
                       dpi = 600)
            plt.show()
            
            plt.figure(figsize=(8, 4))
            #plt.subplot(2,1,2)
            plt.plot(history_tr.index[0:1000], history_tr["accuracy"][0:1000], 'orange', label='Exatidão de teste "source"')
            plt.plot(history_val.index[0:1000], history_val["accuracy"][0:1000], 'b', label='Exatidão de teste "target"')
            #plt.title('Training accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_epochs_accuracy.jpg'.format(src, tgt),
                       dpi = 600)
            plt.show()
            
            # ============================================================================================
            # Plot Confusion Matrix
            # Confusion matrix validate sorce
            pred = clf_branch.predict(data_src_fft)
            pred = [np.argmax(x) for x in pred]
            
            cm = confusion_matrix(label_src, pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                         display_labels=np.unique(label_tgt))
            disp.plot()
            plt.xlabel('Classe predita')
            plt.ylabel('Classe verdadeira')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_confusion matrix source validation DANN.jpg'.format(src, tgt),
                       dpi = 600)
            plt.show()
            
            # Confusion matrix validate target
            pred = clf_branch.predict(data_tgt_fft)
            pred = [np.argmax(x) for x in pred]
            
            cm = confusion_matrix(label_tgt, pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                         display_labels=np.unique(label_tgt))
            disp.plot()
            plt.xlabel('Classe predita')
            plt.ylabel('Classe verdadeira')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_confusion matrix target validation DANN.jpg'.format(src, tgt),
                       dpi = 600)
            plt.show()
            
            # ============================================================================================
            ## Feature Factors 
            clf_branch.summary()
        
            feature_output = Model(clf_branch.input, clf_branch.get_layer('feature').output)

            feture_pred = feature_output(data_src_fft)
            feture_pred.shape

            feture_pred_tgt = feature_output(data_tgt_fft)
            feture_pred_tgt

            # ============================================================================================
            # PCA
            pca = PCA(n_components=7)
            pca_model = pca.fit(np.array(feture_pred))
            Xfactor_train_source = pca_model.transform(np.array(feture_pred))
            Xfactor_train_target = pca_model.transform(np.array(feture_pred_tgt))

            Xfactor_source = pd.DataFrame(Xfactor_train_source).reset_index(drop=True)
            Xfactor_target = pd.DataFrame(Xfactor_train_target).reset_index(drop=True)
            y_train_source = pd.DataFrame(label_src).reset_index(drop=True)
            y_train_source['Condition']='Source'
            Xfactor_source[['Label', 'Condition']]= y_train_source
            y_train_target = pd.DataFrame(label_tgt).reset_index(drop=True)
            y_train_target['Condition']='Target'
            Xfactor_target[['Label', 'Condition']]= y_train_target

            Xfactor_train_global = pd.concat([Xfactor_source, Xfactor_target])
            Xfactor_train_global.reset_index(inplace=True, drop=True)
            Xfactor_train_global

            print(
                "explained variance ratio (first two components): %s"
                % str(pca_model.explained_variance_ratio_)
            )

            # ============================================================================================
            # Plot PCA DANN Feature source
            plt.figure(figsize=(8, 4))
            plt.subplot(1,2,1)
            ax = sns.scatterplot(data=Xfactor_train_global[Xfactor_train_global['Condition']=='Source'],
                            x = 0, y = 1, hue = 'Label', 
                            palette = "tab10")
            ax.legend(loc = 'upper right',
                    bbox_to_anchor=(1.25, 1.02))
            plt.xlabel('comp-1')
            plt.ylabel('comp-2')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_PCA_source_feature.jpg'.format(src, tgt),
                    dpi = 600)

            # Plot PCA DANN Feature source
            plt.figure(figsize=(8, 4))
            plt.subplot(1,2,1)
            ax = sns.scatterplot(data=Xfactor_train_global[Xfactor_train_global['Condition']=='Target'],
                            x = 0, y = 1, hue = 'Label', 
                            palette = "tab10")
            ax.legend(loc = 'upper right',
                    bbox_to_anchor=(1.25, 1.02))
            plt.xlabel('comp-1')
            plt.ylabel('comp-2')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_PCA_target_feature.jpg'.format(src, tgt),
                    dpi = 600)

            # Plot PCA DANN Feature source e target
            plt.figure(figsize=(8, 4))
            plt.subplot(1,2,1)
            ax = sns.scatterplot(data=Xfactor_train_global,
                            x = 0, y = 1, hue = 'Condition', 
                            palette = "tab10")
            ax.legend(loc = 'upper right',
                    bbox_to_anchor=(1.4, 1.02))
            plt.xlabel('comp-1')
            plt.ylabel('comp-2')
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_PCA_conditions_distribution_feature.jpg'.format(src, tgt),
                    dpi = 600)


            fig = px.scatter_3d(Xfactor_train_global, x=0, y=1, z=2,
                        color='Condition')
            fig.show()

            # ============================================================================================
            # TSNE
            aux = np.append(np.array(feture_pred),np.array(feture_pred_tgt),axis=0)
            aux.shape

            tsne = TSNE(n_components=2, verbose=1, random_state=123)
            z = tsne.fit_transform(aux)

            yaux = np.append(label_src,label_tgt)

            df = pd.DataFrame()
            df["y"] = yaux
            df["y2"] = 'Source'
            df.loc[2000:, 'y2'] = 'Target'
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]

            # Plot TSNE feature classes
            plt.figure(figsize=(8, 4))
            plt.subplot(1,2,1)
            ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                            palette=sns.color_palette("hls", 10),
                            data=df)
            ax.legend(loc = 'upper right',
                    bbox_to_anchor=(1.25, 1.02))
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_tSNE_distribution_classes_feature.jpg'.format(src, tgt),
                    dpi = 600)

            # Plot TSNE features soruce e target
            plt.figure(figsize=(8, 4))
            plt.subplot(1,2,1)
            ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y2.tolist(),
                            palette=sns.color_palette("hls", 2),
                            data=df)
            ax.legend(loc = 'upper right',
                    bbox_to_anchor=(1.4, 1.02))
            plt.tight_layout()
            plt.savefig('./imagens/T{}{}_tSNE_distribution_conditions_feature.jpg'.format(src, tgt),
                    dpi = 600)
