from keras import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras import optimizers
from keras.models import load_model
from core.time_until import *

class Model():

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()
        # LSTM层
        self.model.add(LSTM(units=configs['model']['n_lstm_out'],
                       input_shape=(configs['data']['n_step'], configs['data']['n_input'])))
        self.model.add(Dropout(0.2))
        # 全连接层
        self.model.add(Dense(units=configs['data']['n_input']))
        # 激活层
        self.model.add(Activation('softmax'))
        # 查看各层的信息
        self.model.summary()
        # 编译
        self.model.compile(
            optimizer=optimizers.adam_v2.Adam(learning_rate=configs['model']['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        print('[Model] Model Compiled')
        timer.stop()
        return self.model

    def train(self, x, y, epochs, batch_size):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        self.history = self.model.fit(x, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)

        self.model.save('./output/model.hdf5')

        print('[Model] Training Completed. Model saved as model.h5')
        timer.stop()

    def predict(self, test_data):
        predicted = self.model.predict(test_data)
        return predicted