#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:21:13 2022

@author: raugom
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import csv
from datetime import datetime
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

from matplotlib import pyplot

import prepare_data as Prepara
import models as models

# Para que coja la memoria de la GPU de manera progresiva
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PARAMETROS IMPORTANTES
units = 64
batch_size = 128
epochs = 200
intento = 82
umbral = 300
DIM_PRINCIPAL = 60*35 + 60*7 + 2# Dimension linea de datos (5min * 35 sensores + 2 horas)
#DIM_PRINCIPAL = 120*35 + 2# Dimension linea de datos (5min * 35 sensores + 2 horas)

def sum(array):    
    sum = 0    
    for i in array: 
        sum += i    
    return sum

if __name__ == '__main__':
    """The entry point"""
    # set and parse the arguments list
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--v', dest='model', action='store', default='', help='deep model')
    args = p.parse_args()
      
    Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation, dictAct = Prepara.getData()
    
    cvaccuracy = []
       
    model = models.get_biLSTM_H(len(Xtrain), units, Prepara.max_lenght, len(dictAct), DIM_PRINCIPAL)
    model = models.compileModel(model)

    model_checkpoint = ModelCheckpoint(
        "best_model_intento" + str(intento),
        monitor='val_accuracy',
        save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # train the model

    print('Begin training ...')    

    log_dir = "./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                                  
    history = None

    history = model.fit(Xtrain, Ytrain, validation_data=(Xvalidation, Yvalidation), epochs=epochs, batch_size=batch_size, verbose=1,
              callbacks=[model_checkpoint, early_stopping, tensorboard_callback])
    del Xtrain,Ytrain,Xvalidation,Yvalidation
    del model
    #load the saved model

    saved_model = load_model("best_model_intento" + str(intento))

    ####  Grafico de entrenamiento
    pyplot.title('Learning Curves (Accuracy)')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.plot(history.history['accuracy'], color='green', linewidth=2, label='Training accuracy')
    pyplot.plot(history.history['val_accuracy'], color='red', linewidth=2, label='Validation accuracy')
    pyplot.legend()
    # pyplot.show()
    GRAFICO_TRAINING = "training_curve_LSTM_accuracy" + str(intento) + ".svg"
    pyplot.savefig(GRAFICO_TRAINING, format='svg', dpi=1200)
    
    ####  Grafico de perdida
    pyplot.clf()
    pyplot.title('Learning Curves (Loss)')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.plot(history.history['loss'], color='green', linewidth=2, label='Training loss')
    pyplot.plot(history.history['val_loss'], color='red', linewidth=2, label='Validation loss')
    pyplot.legend()
    # pyplot.show()
    GRAFICO_LOSS = "training_curve_LSTM_loss" + str(intento) + ".svg"
    pyplot.savefig(GRAFICO_LOSS, format='svg', dpi=1200)

    # evaluate the model user1
    print('Begin testing ...')
    batch_size = 64
    scores = saved_model.evaluate(Xtest, Ytest, batch_size=batch_size, verbose=1)
  
    print('Report:')
    target_names = sorted(dictAct, key=dictAct.get)
       
    classes = saved_model.predict(Xtest, batch_size=batch_size)
    classes = np.argmax(classes, axis=1)
    
    csvfile = 'test-intento' + str(intento) + '.csv'
    
    counter = 0
    with open(csvfile, "w") as output:
        pd.DataFrame(np.asarray(np.transpose(np.vstack((Ytest,classes))))).to_csv(csvfile)
    
    print(classification_report(list(Ytest), classes, target_names=target_names))
    print('Confusion matrix:')
    labels = list(dictAct.values())
    confusionmatrix = confusion_matrix(list(Ytest), classes, labels)
    print(confusionmatrix)
    resultado_csv = 'results' + str(intento) + '.csv'
    pd.DataFrame(confusionmatrix).to_csv(resultado_csv, index=False, header=False)

    cvaccuracy.append(scores[1] * 100)

    print('{:.2f}% (+/- {:.2f}%)'.format(np.mean(cvaccuracy), np.std(cvaccuracy)))
          
    saved_model.save(os.path.join('/home', 'raul', 'BBDD_Raul', 'Neural', 'modeloLSTM-final_i' + str(intento) + '.h5'))

    # PROCESO DE EVALUACION POSTERIOR PARA COMPROBAR UN RANGO DE TIEMPO PARA LA DETECCION DE LA ACTIVIDAD
    # LEO DATOS DE FICERO CSV
    my_data = genfromtxt(csvfile,delimiter=',')

    # ELIMINO CABECERA Y PRIMERA COLUMNA
    my_data = np.delete(np.delete(my_data,obj=0,axis=1),obj=0,axis=0)

    unique, counts = np.unique(my_data, return_counts = True)

    positives = 0
    negatives = 0
    for result in my_data:
        if(result[0]==result[1]):
            positives+=1
        else:
            negatives+=1

    print("------------------")        
    print("Real results")
    print("------------------")
    print("Number of predicts")
    print(positives+negatives)

    print("Test Accuraccy: ")
    print("------------------")
    print(positives/(positives+negatives))

    counter = 0
    flagALG = False
    confusionMatrix = np.zeros((18,18))

    for result in my_data:
        if(result[0]==result[1]): # CASO CORRECTO
            confusionMatrix[int(result[0])][int(result[1])]+=1
        else: # CASO INCORRECTO
            if(counter-umbral >= 0 and counter+umbral <= len(my_data)-1):
                for memory in np.arange(counter-umbral,counter+umbral+1):
                    if(result[0]==my_data[memory][1]):
                        flagALG = True
                if(flagALG == True):
                    confusionMatrix[int(result[0])][int(result[0])]+=1
                else:
                    confusionMatrix[int(result[0])][int(result[1])]+=1
                flagALG = False
            else:
                confusionMatrix[int(result[0])][int(result[1])]+=1
        counter+=1

    print("------------------")
    print("Corrected results")
    print("------------------")
    print("Number of predicts")
    print(np.sum(confusionMatrix))

    print("Test Accuraccy: ")
    print("------------------")
    print((np.trace(confusionMatrix))/(np.sum(confusionMatrix)))

    print(confusionMatrix)
    pd.DataFrame(np.empty([0,0])).to_csv(resultado_csv, mode='a', index=False, header=False)
    pd.DataFrame(confusionMatrix).to_csv(resultado_csv, mode='a', index=False, header=False)