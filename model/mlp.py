'''=================================================
@Project -> File   ：titanic -> mlp
@Author ：Zhuang Yi
@Date   ：2019/10/30 16:00
=================================================='''

import keras
from numpy import concatenate
from tensorflow.python.keras.layers import Concatenate
from keras import Sequential, Input, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import GRU, Dense, K, Dropout, Activation, LSTM
from data_processing.data_processing import Clean_Data
from util import GRU_MODEL_PATH, MLP_MODEL_PATH
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np

class MY_MLP(object):
    def __init__(self):
        self.sd = Clean_Data()
        self.training_x, self.training_y, self.testing_x = self.sd.split_train_test_mlp()
        self.training_x_1 = self.training_x[:, 0:1]
        self.testing_x_1 = self.testing_x[:, 0:1]
        self.training_x_2 = self.training_x[:, 2:]
        self.testing_x_2 = self.testing_x[:, 2:]
        self.id = list(range(892, 1310))

    def mlp_train(self):
        input_1 = Input(shape=(self.training_x_1.shape[1],), name='f_in')

        input_2 = Input(shape=(self.training_x_2.shape[1],), name='r_in')
        y = Dropout(0.2)(input_2)
        y_out = Dense(output_dim=1, activation='relu')(y)

        concatenated = keras.layers.concatenate([input_1, y_out])
        out = Dense(output_dim=1, activation='sigmoid')(concatenated)

        merged_model = Model(inputs=[input_1, input_2], outputs=[out])

        merged_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        tensorboard = TensorBoard(log_dir='mlp_log')
        checkpoint = ModelCheckpoint(filepath=MLP_MODEL_PATH, monitor='val_loss', mode='auto')
        callback_lists = [tensorboard, checkpoint]

        history = merged_model.fit({'f_in': self.training_x_1, 'r_in': self.training_x_2}, self.training_y, verbose=2, epochs=800, batch_size=24,
                            validation_split=0.2,
                            callbacks=callback_lists)

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

        print('Train Successfully')

    def test_predict(self):
        model = load_model(MLP_MODEL_PATH)
        y_pred = model.predict({'f_in': self.testing_x_1, 'r_in': self.testing_x_2})
        for i in range(len(y_pred)):
            y_pred[i] = np.round(y_pred[i])
        print(y_pred)
        return y_pred

    def write_to_file(self):
        data = self.test_predict()
        dataframe = pd.DataFrame({'PassengerId': self.id, 'Survived': data[:, 0]})
        dataframe.to_csv("result_mlp.csv", index=False, sep=',')


if __name__ == '__main__':
    mlp = MY_MLP()
    mlp.mlp_train()
    mlp.write_to_file()
