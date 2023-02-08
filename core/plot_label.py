import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import  ConnectionPatch


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

def plot_lable1(wave,tp,tp_lable,predict):
    wave = np.array(wave).reshape(2000)
    x = np.arange(0,len(wave),1)
    tp_lable = np.array(tp_lable).reshape(2000)
    predict = np.array(predict).reshape(2000)
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(x,wave,'b')
    ax1.vlines(tp, -0.5, 0.5, colors='r', linestyles='dashed', label='P')
    # axins1 = ax1.inset_axes((0.1, 0.1, 0.4, 0.3))
    # axins1.plot(x, wave, color='b', label='trick', alpha=0.7)
    # axins1.vlines(tp, -0.5, 0.5, colors='r', linestyles='dashed', label='P')
    # zone_and_linked(ax1, axins1, int(tp-50), int(tp+50), x, [wave], 'left')
    plt.legend()
    ax2 = fig.add_subplot(2,1,2)
    ax2.bar(x,height=predict,width=2)
    axins2 = ax2.inset_axes([0.1, 0.2, 0.4, 0.3])
    axins2.bar(x, height=predict, width=2, alpha=0.7)
    zone_and_linked(ax2, axins2, int(tp - 50), int(tp + 50), x, [predict], 'left')
    plt.legend()
    plt.savefig('lable.png')
    plt.show()