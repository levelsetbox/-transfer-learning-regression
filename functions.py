import pandas as pd
import numpy as np
import math 
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import gdal


def pre_process_data(filename, x_start, y_column):
    # 读取数据，是否需要归一化
    # 归一化：对每个特征独立的进行归一化：均值：0，方差：1
    # names = ['B75', 'B3', 'B244', 'B24', 'B187', 'B283', 'B278', 'B132', 'B185', 'B46', 'B158', 'B128', 'B8', 'B171',
    #          'B4',
    #          'B84',
    #          'B236', 'B280', 'B115', 'B72']
    data = pd.read_excel(filename, header=0)
    # Y=data.iloc[:,y_column]
    # X=data.loc[:,names]
    X_dataframe = data.iloc[:, x_start:]
    Y_dataframe = data.iloc[:, y_column]
    X = np.array(X_dataframe)
    Y = np.array(Y_dataframe)
    return data, X_dataframe, Y_dataframe, X, Y

def check_condition(test_index):
    # 检查数组是否满足分组条件：相邻两索引之差<=9，;起始元素<=4或者最末元素>=34；同组内不出现重复的
    # 满足条件：True
    test_group = [math.ceil((i + 1) / 4) for i in test_index]  # 索引所在的组号
    delta = [test_index[i + 1] - test_index[i] for i in range(len(test_index) - 1)]  # 相邻两元素之差
    max_ = max(delta)  # 相邻两元素之差最大值
    if max_ <= 7:
        if len(set(test_group)) == len(test_group):  # 判断是否重组
            return True


def statistics(Y):
    """

    :param Y值:
    :return: Y的统计值
    """
    y_max = Y.max()
    y_min = Y.min()
    y_mean = Y.mean()
    y_median = np.median(Y)
    y_std = Y.std(ddof=1)
    y_cv = y_std / y_mean
    return y_max, y_min, y_mean, y_median, y_std, y_cv


def train_model(model, x_train, y_train, x_test, y_test, return_pred=False):
    """
    返回模型训练结果
    :param model:xgb模型
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return: r2train_xgb, rmsetrain_xgb, rpdtrain_xgb, mretrain_xgb,
 r2test_xgb, rmsetest_xgb, rpdtest_xgb, mretest_xgb
    """
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    r2train_xgb, rmsetrain_xgb, rpdtrain_xgb, mretrain_xgb = metrics_(pred_train, y_train)
    r2test_xgb, rmsetest_xgb, rpdtest_xgb, mretest_xgb = metrics_(pred_test, y_test)
    if return_pred == True:
        return r2train_xgb, rmsetrain_xgb, rpdtrain_xgb, mretrain_xgb, r2test_xgb, rmsetest_xgb, rpdtest_xgb, mretest_xgb, pred_train, pred_test
    else:
        return r2train_xgb, rmsetrain_xgb, rpdtrain_xgb, mretrain_xgb, r2test_xgb, rmsetest_xgb, rpdtest_xgb, mretest_xgb


def metrics_(y_pred, y_true):
    # 输出r^2,rmse,mre,rpd
    y_true = np.array(y_true)
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    rpd = np.std(y_pred, ddof=1) / rmse
    relative_error = np.average(np.abs(y_true - y_pred) / y_true)
    return r2, rmse, rpd, relative_error


def graph(y_train, pred_train, y_test, pred_test):
    # 画真值、预测值散点图（训练集、测试集）
    plt.rcParams['xtick.direction'] = 'in'  # 设置坐标轴刻度朝向为内
    plt.rcParams['ytick.direction'] = 'in'
    font_aixs = {'family': 'Times New Roman',
                 'size': 13,
                 'weight': 'bold'}  # 设置坐标轴名称字体参数：字体、大小、加粗
    font_legend = {'family': 'Times New Roman',  # 设置图例名称字体参数：字体、大小、加粗
                   'size': 14,
                   'weight': 'bold'}
    font_ticks = {'family': 'Times New Roman',  # 设置坐标轴刻度标注字体参数：字体、大小、加粗
                  'size': 13,
                  'weight': 'normal'
                  }
    plt.xlabel('Measured value/(mg·kg$^{-1}$)', font=font_aixs)  # 设置xy轴名称参数
    plt.ylabel('Predicted value/(mg·kg$^{-1}$)', font=font_aixs)
    # plt.xlabel('Measured value(mg/kg)', font=font_aixs)  # 设置xy轴名称参数
    # plt.ylabel('Predicted value(mg/kg)', font=font_aixs)
    color1 = '#00CED1'  # 图例颜色
    color2 = '#DC143C'
    plt.yticks(font=font_ticks)  # 设置坐标轴刻度参数
    plt.xticks(font=font_ticks)
    area = np.pi * 4 ** 2.2  # 散点大小
    X = np.linspace(10, 100)  # 坐标轴范围
    Y = X
    plt.scatter(y_train, pred_train, s=area, edgecolors='blue', c='white', alpha=0.8, label='Calibration', marker='o')
    # edgecolors='blue', c='white'边缘设置颜色，中间为白色，实现空心
    plt.scatter(y_test, pred_test, s=area, edgecolors='darkred', c='white', alpha=0.8, label='Prediction', marker='^')
    plt.plot(X, Y, linewidth='1', color='#000000', label='y=x')
    plt.legend(prop=font_legend)
    # acc_dict_pb = {
    #     'R-CFS-XGB': 'R$^2$$_c$=0.88, RMSE$_c$=2.17, RPD$_c$=2.75,\nR$^2$$_p$=0.65, RMSE$_p$=3.49, RPD$_p$=1.53',
    #     'Log-CFS-PLSR': 'R$^2$$_c$=0.51, RMSE$_c$=4.36, RPD$_c$=1.04,\nR$^2$$_p$=0.33, RMSE$_p$=4.83, RPD$_p$=0.64',
    #     'R-GA-XGB': 'R$^2$$_c$=0.90, RMSE$_c$=1.90, RPD$_c$=3.21,\nR$^2$$_p$=0.79, RMSE$_p$=2.71, RPD$_p$=1.97',
    #     'R-GA-PLSR': 'R$^2$$_c$=0.76, RMSE$_c$=3.04, RPD$_c$=1.81,\nR$^2$$_p$=0.62, RMSE$_p$=3.67, RPD$_p$=1.45'}
    # acc_dict_CU = {
    #     'CR-CFS-XGB': 'R$^2$$_c$=0.71, RMSE$_c$=9.07, RPD$_c$=1.34,\nR$^2$$_p$=0.55, RMSE$_p$=8.38, RPD$_p$=1.16',
    #     'CR-CFS-PLSR': 'R$^2$$_c$=0.91, RMSE$_c$=4.87, RPD$_c$=2.69,\nR$^2$$_p$=0.42, RMSE$_p$=9.50, RPD$_p$=1.00',
    #     'CR-GA-XGB': 'R$^2$$_c$=0.92, RMSE$_c$=4.93, RPD$_c$=2.87,\nR$^2$$_p$=0.71, RMSE$_p$=6.71, RPD$_p$=1.72',
    #     'CR-GA-PLSR': 'R$^2$$_c$=0.78, RMSE$_c$=7.87, RPD$_c$=1.60,\nR$^2$$_p$=0.63, RMSE$_p$=7.64, RPD$_p$=1.32'}
    trans = {
        '17_19_tr': 'R$^2$$_c$=0.78, RMSE$_c$=10.83, RPD$_c$=2.51,\nR$^2$$_p$=0.66, RMSE$_p$=7.23, RPD$_p$=1.76',
        '17_19': 'R$^2$$_c$=0.74, RMSE$_c$=11.78, RPD$_c$=1.93,\nR$^2$$_p$=0.55, RMSE$_p$=8.33, RPD$_p$=1.37',
        '19': 'R$^2$$_c$=0.82, RMSE$_c$=7.30, RPD$_c$=1.95,\nR$^2$$_p$=0.60, RMSE$_p$=7.84, RPD$_p$=1.64'}
    # $^2$$_c$  上标,下标
    # 添加文字
    plt.text(x=40, y=8, s=trans['19'], fontdict=font_legend)
    # x,y文字所放位置
    plt.show()

class image_operation():
    '''
    对影像进行一系列操作:
        读取:read_image;
        写入:write_image
    '''

    def read_image(self, feature_index, filename, feature_select=False):
        # 读取影像，重构数据
        # filename: 读取影像文件名
        # return im_bands, im_proj, im_geotrans, data_list

        data = gdal.Open(filename)  # 打开文件
        im_width = data.RasterXSize  # 读取图像行数
        im_height = data.RasterYSize  # 读取图像列数
        im_bands = data.RasterCount  # 读取图像波段数

        im_geotrans = data.GetGeoTransform()

        im_proj = data.GetProjection()  # 地图投影信息
        im_data = data.ReadAsArray(0, 0, im_width, im_height)  # 此处读取整张图像

        del data
        if feature_select == True:
            im_data = im_data[feature_index, :, :]

        # 重构数据(多波段数据)
        if im_bands > 1:
            fsl_data = im_data.transpose((1, 2, 0))
            data_list = []
            for i in range(fsl_data.shape[0]):
                for j in range(fsl_data.shape[1]):
                    data_list.append(fsl_data[i][j])
            data_list = np.array(data_list)
        else:
            data_list = np.array(im_data)
        return im_bands, im_proj, im_geotrans, data_list, im_data

    def write_image(self, filename, im_proj, im_geotrans, im_data):
        # 写入影像
        # im_proj, im_geotrans, im_data:投影参数,仿射变换参数,数据矩阵
        # filename:写入影像文件名
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        if len(im_data.shape) == 3:  # len(im_data.shape)表示矩阵的维数
            im_bands, im_height, im_width = im_data.shape  # （维数，行数，列数）
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape  # 一维矩阵

        # 创建文件
        driver = gdal.GetDriverByName('GTiff')  # 数据类型必须有，因为要计算需要多大内存空间
        data = driver.Create(filename, im_width, im_height, im_bands, datatype)
        data.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        data.SetProjection(im_proj)  # 写入投影
        if im_bands == 1:
            data.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                data.GetRasterBand(i + 1).WriteArray(im_data[i])
        del data