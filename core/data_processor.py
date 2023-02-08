import os
import random
import numpy as np
import pandas as pd
from obspy import read
from obspy.core import UTCDateTime

class DataLoader():

    def __init__(self, train_file, test_file, cols1, cols2, normal,delta):
        # 读取数据
        # train_data
        self.data_train_X = []
        self.data_train_Y = []
        dataframe = pd.read_csv(train_file)
        wavefile = dataframe.get(cols1).values
        tp = dataframe.get(cols2).values
        for i in range(wavefile.shape[0]):
            f_name = np.char.strip(str(wavefile[i]), '[ ] \'')
            f_dir = os.path.join("data/train", str(f_name))
            f_st = read(f_dir)
            f_tr = f_st[0]
            start_time = f_tr.stats.starttime
            tp_time = np.char.strip(str(tp[i]), '[ ] \'')
            start_time = UTCDateTime(str(start_time))
            tp_time = UTCDateTime(str(tp_time))
            f_y = int((tp_time-start_time)/delta)
            f_wave = f_tr.data[:]
            # 归一化
            if normal:
                f_wave = f_wave/max(abs(f_wave))
            self.data_train_X.append(f_wave)
            self.data_train_Y.append(f_y)
            for i in range(len(f_wave)):
                rand = random.gauss(0, 0.02)
                f_wave[i] = f_wave[i] + rand
            self.data_train_X.append(f_wave)
            self.data_train_Y.append(f_y)
        self.data_train_X = np.array(self.data_train_X)
        self.data_train_Y = np.array(self.data_train_Y)
        # print(self.data_train_X.shape)
        # print(self.data_train_Y.shape)

        # test_data
        dataframe = pd.read_csv(test_file)
        wavefile = dataframe.get(cols1).values
        f_name = np.char.strip(str(wavefile), '[ ] \'')
        f_dir = os.path.join("data/test", str(f_name))
        f_st = read(f_dir)
        f_tr = f_st[0]
        self.predic_start_time = f_tr.stats.starttime
        self.predic_start_time = UTCDateTime(str(self.predic_start_time))
        f_wave = f_tr.data[:]
        # 归一化
        if normal:
            f_wave = f_wave / max(abs(f_wave))
        self.data_test_X = np.array(f_wave)
        # print(self.data_test_X)

    def get_train_data(self, seq_len, window_num):
        data_x = []
        data_y = []
        for i in range(self.data_train_X.shape[0]):
            for j in range(window_num):
                dely = random.randint(self.data_train_Y[i]-seq_len+100,self.data_train_Y[i])
                x = self.data_train_X[i][dely:seq_len+dely]
                y = self.data_train_Y[i] - dely
                data_x.append(x)
                data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_test_data(self, seq_len, cut_start, delta):
        data_x = []
        start=int(cut_start/delta)
        x = self.data_test_X[start:seq_len+start]
        data_x.append(x)
        return np.array(data_x)

