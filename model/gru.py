'''=================================================
@Project -> File   ：titanic -> gru
@Author ：Zhuang Yi
@Date   ：2019/10/24 21:34
=================================================='''

# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import keras
from numpy import concatenate
from tensorflow.python.keras.layers import Concatenate
from keras import Sequential, Input, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import GRU, Dense, K, Dropout, Activation, LSTM
from data_processing.data_processing import Clean_Data
from util import GRU_MODEL_PATH
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np

class MY_GRU(object):
    def __init__(self):
        self.sd = Clean_Data()
        self.training_x, self.training_y, self.testing_x = self.sd.split_train_test()
        self.re_training_x, self.re_testing_x = self.sd.reverse_train_test()

    def gru_train(self):
        input_1 = Input(shape=(self.training_x.shape[1], self.training_x.shape[2]), name='f_in')
        x = GRU(input_dim=1, output_dim=6, return_sequences=True)(input_1)
        x = GRU(6, return_sequences=False)(x)
        x_out = Dense(output_dim=3, activation='relu')(x)

        input_2 = Input(shape=(self.re_training_x.shape[1], self.re_training_x.shape[2]), name='r_in')
        y = GRU(input_dim=1, output_dim=6, return_sequences=True)(input_2)
        y = GRU(6, return_sequences=False)(y)
        y_out = Dense(output_dim=3, activation='relu')(y)

        concatenated = keras.layers.concatenate([x_out, y_out])
        out = Dense(output_dim=1, activation='relu')(concatenated)
        out = Dropout(0.1)(out)

        merged_model = Model(inputs=[input_1, input_2], outputs=[out])



        # model = Sequential()
        # model.add(GRU(input_dim=1, output_dim=7, return_sequences=True))
        # model.add(Dropout(0.2))
        #
        # model.add(GRU(7, return_sequences=False))
        # model.add(Dropout(0.1))
        # model.add(Dense(output_dim=1, activation='sigmoid'))
        merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        tensorboard = TensorBoard(log_dir='gru_log')
        checkpoint = ModelCheckpoint(filepath=GRU_MODEL_PATH, monitor='val_loss', mode='auto')
        callback_lists = [tensorboard, checkpoint]

        history = merged_model.fit({'f_in': self.training_x, 'r_in': self.re_training_x}, self.training_y, verbose=2, epochs=300, batch_size=20,
                            validation_split=0.3,
                            callbacks=callback_lists)

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

        print('Train Successfully')

    def test_predict(self):
        model = load_model(GRU_MODEL_PATH)
        y_pred = model.predict({'f_in': self.testing_x, 'r_in': self.re_testing_x})
        for i in range(len(y_pred)):
            y_pred[i] = np.round(y_pred[i])
        print(y_pred)
        return y_pred

    def write_to_file(self):
        data = self.test_predict()
        dataframe = pd.DataFrame({'Survived': data[:,0]})
        dataframe.to_csv("result.csv", index=False, sep=',')


if __name__ == '__main__':
    gru = MY_GRU()
    gru.gru_train()
    gru.write_to_file()
