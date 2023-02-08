import os

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras import optimizers
import json
import numpy as np
from  core.data_processor import *
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from core.model_create import *
from core.plot_map import *
from core.plot_label import *
import  random
from matplotlib import pyplot

def main():
    #读取参数
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['output_dir']): os.makedirs(configs['model']['output_dir'])
    #读取数据
    data = DataLoader(
        os.path.join('data', configs['data']['train_file']),
        os.path.join('data', configs['data']['test_file']),
        configs['data']['columns1'],
        configs['data']['columns2'],
        configs['data']['normalise'],
        configs['data']['delta']
    )
    # 训练数据
    x_train, y_train = data.get_train_data(configs['data']['n_input'],configs['data']['n_random'])
    x_train = x_train.reshape(-1, configs['data']['n_step'], configs['data']['n_input'])
    y_train = tf.keras.utils.to_categorical(y_train, configs['data']['n_input'])
    #print(x_train.shape)
    #print(y_train.shape)

    model = Model()
    # my_model = model.build_model(configs)
    # plot_model(my_model, to_file='model.png',show_shapes=True)
    #
    # #训练模型
    # model.train(
    # 	x_train,
    # 	y_train,
    # 	configs['training']['epochs'],
    # 	configs['training']['batch_size'])
    #
    # plot_history(model.history)

    # 载入训练好model
    model = load_model('./output/model.hdf5')

    # 测试
    x = []
    y = []
    for i in range(200):
        dely = random.randint(0,1200)
        x_test = data.get_test_data(configs['data']['n_input'],dely*0.05,configs['data']['delta'])
        for i in range(x_test.shape[1]):
            rand=random.gauss(0,0.02)
            x_test[0][i]=x_test[0][i]+rand
        x=np.append(x,dely)
        x_test = x_test.reshape(-1, configs['data']['n_step'], configs['data']['n_input'])
        predict_y = model.predict(x_test)
        y=np.append(y,np.argmax(predict_y))
    x=np.array(x)
    y=np.array(y)
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    plt.rcParams.update({'font.size': 13})
    # ax1.text(3, 3, 'c)')
    plt.xlim(0,60)
    plt.ylim(0,60)
    ax1.set_xlabel('window_start_time/s')
    ax1.set_ylabel('tp/s')
    ax1.scatter(x*0.05, y*0.05,s=5, c='k')
    #ax1.legend()
    plt.savefig('./output/error.png')


if __name__ == '__main__':
    main()