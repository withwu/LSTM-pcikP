# LSTM-pickP
[LSTM neural network for automatic P phase picking](https://github.com/withwu/LSTM-pickP) <br />
作者：吴为治<br />
<br />
##1.运行基础 <br />
程序运行基于以下库：<br />
[tensorflow](https://tensorflow.google.cn/install?hl=zh-cn) <br />
[keras](https://keras.io/) <br />
[obspy](https://github.com/obspy/obspy) <br />
[pandas](https://pandas.pydata.org/) <br />
[numpy](https://numpy.org/install/) <br />
[matplotlib](https://matplotlib.org/3.1.1/users/installing.html) <br />
<br />
##2.文件说明：<br />
core文件夹：<br />数据读入、预处理、模型建立以及作图程序。<br />
data文件夹：<br />train文件夹为训练地震数据集；test文件夹为测试地震数据集；
    train.csv文件为训练集文件目录及人工拾取到时；test.csv文件为测试集文件目录。<br />
image文件夹：国内用户会存在该readme图片不显示情况，可以下载程序包文件解压查看。<br />
output文件夹: 训练好的模型以及结果图输出路径<br />
config.json文件：程序运行时所需参数<br />
<br />
##3.程序运行（ps：国内用户图片有不显示情况，可以按网上教程修改host或者下载程序包查看image文件夹）<br />
运行run.py<br />
模型生成<br />![img.png](./image/img.jpg) <br />
模型训练<br />![img_1.png](./image/img_1.jpg) <br />
结果输出<br />![img_2.png](./image/img_2.jpg) <br />
拾取效果<br />![img_3.png](./image/img_3.png) <br />

运行run2.py<br />
生成不同截图时窗可靠性分析图<br /> ![img_4.png](./image/img_4.png) <br />
