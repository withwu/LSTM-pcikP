import os
import random
import pandas as pd
from obspy import read
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

dataframe = pd.read_csv('../data/train.csv')
wavefile = dataframe.get('wavefile').values
tp = dataframe.get('tp').values
f_name = np.char.strip(str(wavefile[4]), '[ ] \'')
f_dir = os.path.join("../data/train", str(f_name))
f_st = read(f_dir)
f_tr = f_st[0]
start_time = f_tr.stats.starttime
tp_time = np.char.strip(str(tp[4]), '[ ] \'')
start_time = UTCDateTime(str(start_time))
tp_time = UTCDateTime(str(tp_time))
f_y = int((tp_time-start_time)/0.05)
f_wave = f_tr.data[:]
# 归一化
f_wave = f_wave/max(abs(f_wave))


x = np.arange(0,len(f_wave),1)
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(x*0.05,f_wave,'lightgray')
ax1.vlines(f_y*0.05, -0.3, 0.3, colors='k', linestyles='dashed', label='P')
ax1.vlines((f_y-1200)*0.05, -0.3, 0.3, colors='k', linestyles='dotted', label='tp-60s')
ax1.vlines((f_y-340)*0.05, -1.1, 1.1, colors='k', linestyles='dashdot', label='window')
ax1.vlines((f_y-340+1200)*0.05, -1.1, 1.1, colors='k', linestyles='dashdot')
ax1.hlines(-1.1, (f_y-340)*0.05,(f_y-340+1200)*0.05, colors='k', linestyles='dashdot')
ax1.hlines(1.1, (f_y-340)*0.05,(f_y-340+1200)*0.05, colors='k', linestyles='dashdot')
plt.ylim(-1.5,1.5)
plt.xlabel('t/delta')
ax1.legend()
#plt.show()
plt.savefig('./cut_window.png')
