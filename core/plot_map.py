import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import  ConnectionPatch

def plot_history(history):
    fig = plt.figure(figsize=(6, 5))
    plt.rcParams.update({'font.size': 10})
    #plt.title('Training accuracy and loss',fontsize=30)
    acc = history.history['accuracy']
    loss = history.history['loss']
    x = range(1, len(acc) + 1)
    ax1 = fig.add_subplot(111)
    ax1.plot(x, acc, 'k', label='training accuracy')
    #plt.ylim(0,1.2)
    ax1.legend(bbox_to_anchor=(1,0.9))
    ax1.set_ylabel('accuracy values')
    ax1.set_xlabel('epochs')
    ax2 = ax1.twinx()
    ax2.plot(x, loss, 'gray',linestyle = "-.",label='training loss')
    ax2.legend(bbox_to_anchor=(1,0.82))
    ax2.set_ylabel('loss values')
    #plt.show()
    plt.savefig('./output/history.png')

def plot_result(wave,tp_lable,n_pridic,n_input):
    wave = np.array(wave).reshape(n_input)
    x = 0.05*np.arange(0,len(wave),1)
    tp_lable = np.array(tp_lable).reshape(n_input)
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 13})
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(x,wave,'lightgray')
    ax1.vlines(n_pridic*0.05, -0.3, 0.3, colors='k', linestyles='dashed', label='predict P')
    ax1.set_xlabel('t/s')
    ax1.legend()
    ax2 = fig.add_subplot(2,1,2)
    ax2.bar(x,tp_lable,width=1,color='k')
    #ax2.legend()
    ax2.set_xlabel('t/s')
    ax2.set_ylabel('softmax')
    #plt.show()
    plt.savefig('./output/result.png')

def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.0, y_ratio=0.0):

    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)

def plot_result2(wave,tp):
    wave = np.array(wave).reshape(1200)
    x = 0.05*np.arange(0,len(wave),1)
    fig = plt.figure(figsize=(7, 4.5))
    plt.rcParams.update({'font.size': 13})
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('t/s')
    ax1.plot(x,wave,'lightgray')
    ax1.text(0,-0.6,'c)')
    ax1.vlines(tp*0.05, -0.3, 0.3, colors='k', linestyles='dashed', label='P')
    axins1 = ax1.inset_axes((0.1, 0.6, 0.4, 0.3))
    axins1.plot(x, wave, 'lightgray', label='trick', alpha=0.7)
    axins1.vlines(tp*0.05, -0.3, 0.3, colors='k', linestyles='dashed', label='P')
    zone_and_linked(ax1, axins1, int(tp-20), int(tp+20), x, [wave], 'left')
    plt.legend()
    plt.savefig('./output/result2.png')