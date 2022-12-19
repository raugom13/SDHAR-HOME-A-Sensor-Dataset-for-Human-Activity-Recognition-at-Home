#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:49:47 2022

@author: raugom
"""

from keras.layers import Dense, LSTM, GRU, Bidirectional, Concatenate, Dropout, Input, RepeatVector, Layer, SimpleRNN
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import adam_v2 as Adam
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tensorflow.shape(x)[-1]
        positions = tensorflow.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def get_LSTM(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    model = Sequential(name='LSTM')
    #model.add(Input(shape=(1,167 ), dtype=tensorflow.float32,))
    #model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=False))
    #model.add(Dropout(0.1))
    #model.add(LSTM(output_dim, kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), bias_regularizer=l2(0.000001)))
    #model.add(LSTM(input_dim))
    #model.add(RepeatVector(1) for i in range(5))
    #model.add(layers.BatchNormalization())
    #model.add(LSTM(output_dim,activation='relu',input_shape=(1,167))) # 5*35 + 2
    model.add(LSTM(output_dim,activation='relu',kernel_regularizer=l2(0.000001), 
                   recurrent_regularizer=l2(0.000001), bias_regularizer=l2(0.000001),
                   input_shape=(1,DIM_PRINCIPAL))) # 5*35 + 2
    #model.add(layers.BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(no_activities, activation='softmax'))
    return model

def get_LSTM_H(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):

    inputs = keras.Input(shape=(DIM_PRINCIPAL,), dtype=tensorflow.float32, name='inputsLayer')

    input1, input2 = tensorflow.split(inputs, [2,(DIM_PRINCIPAL-2)], axis=1, name='splitLayer')
    
    list_Frames_3D = [layers.RepeatVector(30)(input2)]
    
    LSTM = layers.LSTM(output_dim, activation='relu', input_shape = (5, DIM_PRINCIPAL-2), name='LSTM')(list_Frames_3D)
    
    Dropout_LSTM = layers.Dropout(0.1, name='Dropout_LSTM')(LSTM)

    Concatenate = layers.concatenate([input1,Dropout_LSTM], axis=1)
    
    ##############################################################################
    #                     samples                 118844                         #
    #   NEURONAS = --------------------- = --------------------- = 216.47 -> 220 #
    #               x(Entradas+Salidas)          3(167+16)                       #
    ##############################################################################
    
    Densa = layers.Dense(220, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    
    Dropout_Densa = layers.Dropout(0.3, name='Dropout_Densa')(Densa)
    
    # Finalmente construimos la salida
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Densa)

    model = keras.Model(inputs=inputs, outputs=Output, name="modeloDENSO")
    
    return model

def get_Transformer(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    """
    ventana = int((DIM_PRINCIPAL - 2)/(35))
    
    inputs = keras.Input(shape=(DIM_PRINCIPAL,), dtype=tensorflow.float32, name='inputsLayer')

    input1, input2 = tensorflow.split(inputs, [2,(DIM_PRINCIPAL-2)], axis=1, name='splitLayer')
    
    list_Frames_3D = [layers.RepeatVector(ventana)(input2)]
    
    Transformer = TransformerBlock(DIM_PRINCIPAL-2, 16, 64)(list_Frames_3D)
    
    #Dropout_LSTM = layers.Dropout(0.2, name='Dropout_LSTM')(BI_LSTM)
    BatchNormalization = layers.BatchNormalization()(Transformer)

    Concatenate = layers.concatenate([input1,BatchNormalization], axis=1)
    
    ##############################################################################
    #                     samples                 118844                         #
    #   NEURONAS = --------------------- = --------------------- = 216.47 -> 220 #
    #               x(Entradas+Salidas)          3(167+16)                       #
    ##############################################################################
    
    Densa = layers.Dense(4700, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    
    Dropout_Densa = layers.Dropout(0.2, name='Dropout_Densa')(Densa)
    
    # Finalmente construimos la salida
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Densa)

    model = keras.Model(inputs=inputs, outputs=Output, name="modeloDENSO")
    """
    embed_dim = 42  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 42  # Hidden layer size in feed forward network inside transformer
    ventana = int((DIM_PRINCIPAL - 2)/(42))
    inputs = keras.Input(shape=(DIM_PRINCIPAL,), dtype=tensorflow.float32, name='inputsLayer')

    input1, input2 = tensorflow.split(inputs, [2,(DIM_PRINCIPAL-2)], axis=1, name='splitLayer')
    #list_Frames_3D = [layers.RepeatVector(ventana)(input2)]
    embedding_layer = TokenAndPositionEmbedding(DIM_PRINCIPAL-2, DIM_PRINCIPAL-2, embed_dim)
    x = embedding_layer(input2)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    Concatenate = layers.concatenate([input1,x], axis=1)
    
    Densa = layers.Dense(100, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    
    Dropout_Densa = layers.Dropout(0.2, name='Dropout_Densa')(Densa)
    
    # Finalmente construimos la salida
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Densa)
    
    model = keras.Model(inputs=inputs, outputs=Output, name="modeloDENSO")
    
    return model

def get_Sensors_Beacons_Hour(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    
    ventana = int((DIM_PRINCIPAL - 2)/(42))
    
    inputs = keras.Input(shape=(DIM_PRINCIPAL,), dtype=tensorflow.float32, name='inputsLayer')

    hora, sensores, beacons = tensorflow.split(inputs, [2,(ventana*35),(ventana*7)], axis=1, name='splitLayer')
    
    
    # RED PARA SENSORES
    list_Frames_3D_S = [layers.RepeatVector(ventana)(sensores)]

    BI_LSTM_S = layers.Bidirectional(LSTM(output_dim, activation='relu', input_shape = (ventana, ventana*35),
                                        kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), 
                                        bias_regularizer=l2(0.000001), name='LSTM_S'))(list_Frames_3D_S)
          
    Dropout_LSTM_S = layers.Dropout(0.2, name='Dropout_LSTM_S')(BI_LSTM_S)
    BatchNormalization_S = layers.BatchNormalization()(Dropout_LSTM_S)
    
    #RED PARA BEACONS
    list_Frames_3D_B = [layers.RepeatVector(ventana)(beacons)]

    BI_LSTM_B = layers.Bidirectional(LSTM(output_dim, activation='relu', input_shape = (ventana, ventana*7),
                                        kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), 
                                        bias_regularizer=l2(0.000001), name='LSTM_B'))(list_Frames_3D_B)
          
    Dropout_LSTM_B = layers.Dropout(0.2, name='Dropout_LSTM_B')(BI_LSTM_B)
    BatchNormalization_B = layers.BatchNormalization()(Dropout_LSTM_B)
    

    Concatenate = layers.concatenate([hora,BatchNormalization_S,BatchNormalization_B], axis=1)
    
    ##############################################################################
    #                     samples                 118844                         #
    #   NEURONAS = --------------------- = --------------------- = 216.47 -> 220 #
    #               x(Entradas+Salidas)          3(167+16)                       #
    ##############################################################################
    
    Densa1 = layers.Dense(4400, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    #Dropout_Densa = layers.Dropout(0.4, name='Dropout_Densa')(Densa1)
    Densa2 = layers.Dense(2200, activation='relu', use_bias=False,name='denseLayer2')(Densa1)
    
    Dropout_Final = layers.Dropout(0.4, name='Dropout_Final')(Densa2)
    
    # Finalmente construimos la salida
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Final)

    model = keras.Model(inputs=inputs, outputs=Output, name="modeloDENSO")
    
    return model

def get_Sensors(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    
    ventana = int((DIM_PRINCIPAL)/(35))
    
    inputs = keras.Input(shape=(DIM_PRINCIPAL,), dtype=tensorflow.float32, name='inputsLayer')
     
    # RED PARA SENSORES
    list_Frames_3D_S = [layers.RepeatVector(ventana)(inputs)]

    BI_LSTM_S = layers.Bidirectional(LSTM(output_dim, activation='relu', input_shape = (ventana, ventana*35),
                                        kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), 
                                        bias_regularizer=l2(0.000001), name='LSTM_S'))(list_Frames_3D_S)
          
    Dropout_LSTM_S = layers.Dropout(0.2, name='Dropout_LSTM_S')(BI_LSTM_S)
    BatchNormalization_S = layers.BatchNormalization()(Dropout_LSTM_S)
    
    ##############################################################################
    #                     samples                 118844                         #
    #   NEURONAS = --------------------- = --------------------- = 216.47 -> 220 #
    #               x(Entradas+Salidas)          3(167+16)                       #
    ##############################################################################
    
    Densa1 = layers.Dense(4400, activation='relu', use_bias=False,name='denseLayer1')(BatchNormalization_S)
    #Dropout_Densa = layers.Dropout(0.4, name='Dropout_Densa')(Densa1)
    Densa2 = layers.Dense(2200, activation='relu', use_bias=False,name='denseLayer2')(Densa1)
    
    Dropout_Final = layers.Dropout(0.4, name='Dropout_Final')(Densa2)
    
    # Finalmente construimos la salida
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Final)

    model = keras.Model(inputs=inputs, outputs=Output, name="modeloDENSO")
    
    return model

def get_Beacons(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    
    ventana = int((DIM_PRINCIPAL)/(7))
    
    inputs = keras.Input(shape=(DIM_PRINCIPAL,), dtype=tensorflow.float32, name='inputsLayer')
    
    #RED PARA BEACONS
    list_Frames_3D_B = [layers.RepeatVector(ventana)(inputs)]

    BI_LSTM_B = layers.Bidirectional(LSTM(output_dim, activation='relu', input_shape = (ventana, ventana*7),
                                        kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), 
                                        bias_regularizer=l2(0.000001), name='LSTM_B'))(list_Frames_3D_B)
          
    Dropout_LSTM_B = layers.Dropout(0.2, name='Dropout_LSTM_B')(BI_LSTM_B)
    BatchNormalization_B = layers.BatchNormalization()(Dropout_LSTM_B)
    
    ##############################################################################
    #                     samples                 118844                         #
    #   NEURONAS = --------------------- = --------------------- = 216.47 -> 220 #
    #               x(Entradas+Salidas)          3(167+16)                       #
    ##############################################################################
    
    Densa1 = layers.Dense(4400, activation='relu', use_bias=False,name='denseLayer1')(BatchNormalization_B)
    #Dropout_Densa = layers.Dropout(0.4, name='Dropout_Densa')(Densa1)
    Densa2 = layers.Dense(2200, activation='relu', use_bias=False,name='denseLayer2')(Densa1)
    
    Dropout_Final = layers.Dropout(0.4, name='Dropout_Final')(Densa2)
    
    # Finalmente construimos la salida
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Final)

    model = keras.Model(inputs=inputs, outputs=Output, name="modeloDENSO")
    
    return model
    
def get_biLSTM_H(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    
    #ventana = int((DIM_PRINCIPAL - 2)/(35))
    ventana = int((DIM_PRINCIPAL-2)/(42))
    
    inputs = keras.Input(shape=(DIM_PRINCIPAL,), dtype=tensorflow.float32, name='inputsLayer')
    Dropout_Input = layers.Dropout(0.6, name='Dropout_Input')(inputs)
    #hora, sensores, beacons = tensorflow.split(inputs, [2,(ventana*35),(ventana*7)], axis=1, name='splitLayer')
    #sensores, beacons = tensorflow.split(inputs, [(ventana*35),(ventana*7)], axis=1, name='splitLayer')
    hora, sensores = tensorflow.split(Dropout_Input, [2,(DIM_PRINCIPAL-2)], axis=1, name='splitLayer')
    
    # RED PARA SENSORES
    
    list_Frames_3D_S = [layers.RepeatVector(ventana)(sensores)]

    BI_LSTM_S = layers.Bidirectional(SimpleRNN(output_dim, activation='relu', input_shape = (ventana, DIM_PRINCIPAL-2),
                                        kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), 
                                        bias_regularizer=l2(0.000001), name='LSTM_S'))(list_Frames_3D_S)
          
    Dropout_LSTM_S = layers.Dropout(0.6, name='Dropout_LSTM_S')(BI_LSTM_S)
    BatchNormalization_S = layers.BatchNormalization()(Dropout_LSTM_S)
    
    #RED PARA BEACONS
    """
    list_Frames_3D_B = [layers.RepeatVector(int((DIM_PRINCIPAL)/(7)))(beacons)]

    BI_LSTM_B = layers.Bidirectional(LSTM(output_dim, activation='relu', input_shape = (ventana, ventana*7),
                                        kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), 
                                        bias_regularizer=l2(0.000001), name='LSTM_B'))(list_Frames_3D_B)
          
    Dropout_LSTM_B = layers.Dropout(0.2, name='Dropout_LSTM_B')(BI_LSTM_B)
    BatchNormalization_B = layers.BatchNormalization()(Dropout_LSTM_B)
    """
    
    Concatenate = layers.concatenate([hora,BatchNormalization_S], axis=1)
    #Concatenate = layers.concatenate([hora,BatchNormalization_S,BatchNormalization_B], axis=1)
    #Concatenate = layers.concatenate([BatchNormalization_S,BatchNormalization_B], axis=1)
    
    ##############################################################################
    #                     samples                 118844                         #
    #   NEURONAS = --------------------- = --------------------- = 216.47 -> 220 #
    #               x(Entradas+Salidas)          3(167+16)                       #
    ##############################################################################
    
    Densa1 = layers.Dense(8000, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    #Densa1 = layers.Dense(8800, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    #Dropout_Densa = layers.Dropout(0.5, name='Dropout_Densa')(Densa1)
    #Densa2 = layers.Dense(4400, activation='relu', use_bias=False,name='denseLayer2')(Dropout_Densa)
    
    Dropout_Final = layers.Dropout(0.6, name='Dropout_Final')(Densa1)
    
    # Finalmente construimos la salida
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Final)

    model = keras.Model(inputs=inputs, outputs=Output, name="modeloDENSO")
    
    return model

def get_biLSTM_H_with_attention(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    
    #ventana = int((DIM_PRINCIPAL - 2)/(35))
    ventana = int((DIM_PRINCIPAL-2)/(42))
    
    inputs = keras.Input(shape=(DIM_PRINCIPAL,), dtype=tensorflow.float32, name='inputsLayer')
    Dropout_Input = layers.Dropout(0.6, name='Dropout_Input')(inputs)
    #hora, sensores, beacons = tensorflow.split(inputs, [2,(ventana*35),(ventana*7)], axis=1, name='splitLayer')
    #sensores, beacons = tensorflow.split(inputs, [(ventana*35),(ventana*7)], axis=1, name='splitLayer')
    hora, sensores = tensorflow.split(Dropout_Input, [2,(DIM_PRINCIPAL-2)], axis=1, name='splitLayer')
       
    list_Frames_3D_S = [layers.RepeatVector(ventana)(sensores)]

    BI_LSTM_S = layers.Bidirectional(SimpleRNN(output_dim, activation='relu', input_shape = (ventana, DIM_PRINCIPAL-2),
                                          kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), 
                                          bias_regularizer=l2(0.000001), return_sequences=True, name='LSTM_S'))(list_Frames_3D_S)
    
    Dropout_LSTM_S = layers.Dropout(0.6, name='Dropout_LSTM_S')(BI_LSTM_S)
    
    attention_layer = attention()(Dropout_LSTM_S)
    
    Dropout_attention = layers.Dropout(0.6, name='Dropout_attention')(attention_layer)
    BatchNormalization_S = layers.BatchNormalization()(Dropout_attention)
    
    Concatenate = layers.concatenate([hora,BatchNormalization_S], axis=1)
    
    Densa1 = layers.Dense(8000, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    #Densa1 = layers.Dense(8800, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    #Dropout_Densa = layers.Dropout(0.5, name='Dropout_Densa')(Densa1)
    #Densa2 = layers.Dense(4400, activation='relu', use_bias=False,name='denseLayer2')(Dropout_Densa)
    
    Dropout_Final = layers.Dropout(0.6, name='Dropout_Final')(Densa1)
    
    # Finalmente construimos la salida
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Final)

    model = keras.Model(inputs=inputs, outputs=Output, name="modeloDENSO")
    
    return model

def get_GRU_H_with_attention(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    
    #ventana = int((DIM_PRINCIPAL - 2)/(35))
    ventana = int((DIM_PRINCIPAL-2)/(42))
    
    inputs = keras.Input(shape=(DIM_PRINCIPAL,), dtype=tensorflow.float32, name='inputsLayer')
    Dropout_Input = layers.Dropout(0.6, name='Dropout_Input')(inputs)
    #hora, sensores, beacons = tensorflow.split(inputs, [2,(ventana*35),(ventana*7)], axis=1, name='splitLayer')
    #sensores, beacons = tensorflow.split(inputs, [(ventana*35),(ventana*7)], axis=1, name='splitLayer')
    hora, sensores = tensorflow.split(Dropout_Input, [2,(DIM_PRINCIPAL-2)], axis=1, name='splitLayer')
       
    list_Frames_3D_S = [layers.RepeatVector(ventana)(sensores)]
    
    GRU_S = layers.Bidirectional(GRU(output_dim, activation='relu', input_shape = (ventana, DIM_PRINCIPAL-2),
                                            kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), 
                                            bias_regularizer=l2(0.000001), return_sequences=True, name='GRU_S'))(list_Frames_3D_S)
    
    Dropout_LSTM_S = layers.Dropout(0.6, name='Dropout_LSTM_S')(GRU_S)
    
    attention_layer = attention()(Dropout_LSTM_S)
    
    Dropout_attention = layers.Dropout(0.6, name='Dropout_attention')(attention_layer)
    BatchNormalization_S = layers.BatchNormalization()(Dropout_attention)
    
    Concatenate = layers.concatenate([hora,BatchNormalization_S], axis=1)
    
    Densa1 = layers.Dense(8000, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    #Densa1 = layers.Dense(8800, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    #Dropout_Densa = layers.Dropout(0.5, name='Dropout_Densa')(Densa1)
    #Densa2 = layers.Dense(4400, activation='relu', use_bias=False,name='denseLayer2')(Dropout_Densa)
    
    Dropout_Final = layers.Dropout(0.6, name='Dropout_Final')(Densa1)
    
    # Finalmente construimos la salida
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Final)

    model = keras.Model(inputs=inputs, outputs=Output, name="modeloDENSO")
    
    return model

def get_doublebiLSTM_H(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    
    ventana = int((DIM_PRINCIPAL - 2)/(35))
    
    inputs = keras.Input(shape=(DIM_PRINCIPAL,), dtype=tensorflow.float32, name='inputsLayer')

    input1, input2 = tensorflow.split(inputs, [2,(DIM_PRINCIPAL-2)], axis=1, name='splitLayer')
    
    list_Frames_3D = [layers.RepeatVector(ventana)(input2)]

    BI_LSTM = layers.Bidirectional(LSTM(output_dim, activation='relu', input_shape = (ventana, DIM_PRINCIPAL-2),
                                        kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), 
                                        bias_regularizer=l2(0.000001), name='LSTM'))(list_Frames_3D)
          
    Dropout_LSTM = layers.Dropout(0.2, name='Dropout_LSTM')(BI_LSTM)
    BatchNormalization = layers.BatchNormalization()(Dropout_LSTM)

    Concatenate = layers.concatenate([input1,BatchNormalization], axis=1)
    
    ##############################################################################
    #                     samples                 118844                         #
    #   NEURONAS = --------------------- = --------------------- = 216.47 -> 220 #
    #               x(Entradas+Salidas)          3(167+16)                       #
    ##############################################################################
    
    Densa1 = layers.Dense(4400, activation='relu', use_bias=False,name='denseLayer1')(Concatenate)
    #Dropout_Densa = layers.Dropout(0.4, name='Dropout_Densa')(Densa1)
    Densa2 = layers.Dense(2200, activation='relu', use_bias=False,name='denseLayer2')(Densa1)
    
    Dropout_Final = layers.Dropout(0.4, name='Dropout_Final')(Densa2)
    
    # Finalmente construimos la salida
    Output = layers.Dense(no_activities, activation='softmax', name='Output')(Dropout_Final)

    model = keras.Model(inputs=inputs, outputs=Output, name="modeloDENSO")
    
    return model


def get_biLSTM(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    model = Sequential(name='biLSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(Bidirectional(LSTM(output_dim)))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_Ensemble2LSTM(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    model1 = Sequential()
    #model1.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model1.add(Bidirectional(LSTM(output_dim,activation='relu',kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), bias_regularizer=l2(0.000001),input_shape=(1,DIM_PRINCIPAL)))) # 5*33 + 2

    model2 = Sequential()
    #model2.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model2.add(LSTM(output_dim,activation='relu',kernel_regularizer=l2(0.000001), recurrent_regularizer=l2(0.000001), bias_regularizer=l2(0.000001),input_shape=(1,DIM_PRINCIPAL))) # 5*33 + 2

    model = Sequential(name='Ensemble2LSTM')
    model.add(Concatenate([model1, model2], mode='concat'))
    model.add(Concatenate())
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_CascadeEnsembleLSTM(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    model1 = Sequential()
    model1.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model1.add(Bidirectional(LSTM(output_dim, return_sequences=True)))

    model2 = Sequential()
    model2.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model2.add(LSTM(output_dim, return_sequences=True))

    model = Sequential(name='CascadeEnsembleLSTM')
    model.add(Concatenate([model1, model2], mode='concat'))
    model.add(LSTM(output_dim))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_CascadeLSTM(input_dim, output_dim, max_lenght, no_activities, DIM_PRINCIPAL):
    model = Sequential(name='CascadeLSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(Bidirectional(LSTM(output_dim, return_sequences=True)))
    model.add(LSTM(output_dim))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def compileModel(model):
    #optimizer = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
    optimizer = keras.optimizers.Adam(learning_rate=0.00001, decay=1e-6)
    #optimizer = SGD(lr=0.01, momentum=0.9, decay=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model
