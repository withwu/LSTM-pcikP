import os
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
    my_model = model.build_model(configs)
    # plot_model(my_model, to_file='model.png',show_shapes=True)

    #训练模型
    model.train(
    	x_train,
    	y_train,
    	configs['training']['epochs'],
    	configs['training']['batch_size'])

    plot_history(model.history)

    # 载入训练好model
    # model = load_model('./output/model.hdf5')

    ##测试
    x_test = data.get_test_data(configs['data']['n_input'],configs['data']['cut_test_dely'],configs['data']['delta'])
    # 加噪声
    x_test= np.array(x_test)
    print(x_test.shape)
    for i in range(x_test.shape[1]):
        rand=random.gauss(0,0.02)
        x_test[0][i]=x_test[0][i]+rand
    # print(x_test)
    x_test = x_test.reshape(-1, configs['data']['n_step'], configs['data']['n_input'])
    predict_y = model.predict(x_test)
    #print(predict_y)
    print('tp-time:')
    print(0.05*np.argmax(predict_y))
    print('predict tp:')
    print(data.predic_start_time+(np.argmax(predict_y)+configs['data']['cut_test_dely'])/configs['data']['n_input'])

    ##结果图
    plot_result(x_test,predict_y,np.argmax(predict_y),configs['data']['n_input'])
    plot_result2(x_test,np.argmax(predict_y))


if __name__ == '__main__':
    main()