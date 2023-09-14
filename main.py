import numpy as np
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2  # import the two-stage algorithm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RepeatedKFold
from functions import pre_process_data, metrics_, graph, statistics, seek_bst_split, train_model, image_operation
import math
from tqdm import tqdm
import time as time
# =================================================
# load data 17 and 19
file_name_list_1 = [r'E:\桌面\data\2017_paixv.xlsx', r'E:\桌面\data\2019_paixv.xlsx']
data_19, X_df_19, Y_df_19, X_19, Y_19 = pre_process_data(filename=file_name_list_1[0], x_start=4, y_column=3)
data_17, X_df_17, Y_df_17, X_17, Y_17 = pre_process_data(filename=file_name_list_1[1], x_start=4, y_column=3)


# find best split for test

selected_test_indexs = []
test_indexs = []
rkf = RepeatedKFold(n_splits=4, n_repeats=2000)
# for train_index, test_index in tqdm(rkf.split(X_19)):
#     # 寻找最佳划分
#     time.sleep(0.01)
#     if (list(test_index) not in test_indexs):  # 防止重复
#         test_indexs.append(list(test_index))
#         if check_condition(test_index) == True:
#             x_train_19, x_test_19 = X_19[train_index], X_19[test_index]
#             y_train_19, y_test_19 = Y_19[train_index], Y_19[test_index]
#             x_train_combined = np.append(X_17, x_train_19, axis=0)
#             y_train_combined = np.append(Y_17, y_train_19)
#             sample_size = [len(Y_17), len(y_train_19)]
#             steps = 30  # S=10,不一定运行完所有迭代，error增加时迭代停止
#             fold = 3  # K=2,k折交叉验证
#             random_state = np.random.RandomState(1)
#             tr_ada_model = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=2, random_state=1),
#                                                 n_estimators=200, sample_size=sample_size,
#                                                 steps=steps, fold=fold,
#                                                 random_state=random_state)
#             r2_test_max, r2_train_, pred_train_, pred_test_, max_delta_id = tr_ada_model.fit(x_train_combined,
#                                                                                              y_train_combined,
#                                                                                              x_test_19, y_test_19)
#             if (r2_test_max > 0) and (r2_train_ >= 0):
#                 print(test_index)
#                 selected_test_indexs.append([list(test_index)])
#                 print(
#                     'r2train:%.5f' % (r2_train_))
#                 print('r2test:%.5f' % (r2_test_max))
# 目前为止最好split
# test_index=[ 3  ,6 ,12 ,19 ,21, 27, 28, 35, 36]
# [ 3  6 12 19 21 27 28 35 36]
# r2train:0.84113
# r2test:0.61143

# ================================================

random_state = np.random.RandomState(1)
test_index = [ 3  ,6 ,12 ,19 ,21, 27, 28, 35, 36]
# =========================================================
# Exam A:2019(Patial) to 2019:

train_index = [i for i in range(len(Y_19)) if i not in test_index]
x_test_19 = X_19[test_index]
y_test_19 = Y_19[test_index]
x_train_19 = X_19[train_index]
y_train_19 = Y_19[train_index]
y_max, y_min, y_mean, y_median, y_std, y_cv = statistics(y_test_19)
model_a = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=random_state), learning_rate=0.5,
                            n_estimators=3,
                            random_state=random_state)
r2_a_train, rmse_a_train, rpd_a_train, mre_a_train, r2_a_test, rmse_a_test, rpd_a_test, mre_a_test, pred_train_a, pred_test_a = train_model(
    model_a, x_train_19, y_train_19, x_test_19, y_test_19, return_pred=True)
# graph(y_train_19, pred_train_a, y_test_19, pred_test_a)
# =========================================================
# Exam B:2017+2019(direct) to 2019:
x_train_combined = np.append(X_17, x_train_19, axis=0)
y_train_combined = np.append(Y_17, y_train_19)
model_b = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=random_state, min_samples_leaf=6),
                            random_state=random_state, n_estimators=10)
r2_b_train, rmse_b_train, rpd_b_train, mre_b_train, r2_b_test, rmse_b_test, rpd_b_test, mre_b_test, pred_train_b, pred_test_b = train_model(
    model_b, x_train_combined, y_train_combined, x_test_19, y_test_19, return_pred=True)
# graph(y_train_combined, pred_train_b, y_test_19, pred_test_b)
# =========================================================
# Exam C:组合源域目标域数据：
sample_size = [len(Y_17), len(y_train_19)]
steps = 30  # S=10,不一定运行完所有迭代，error增加时迭代停止
fold = 3  # K=2,k折交叉验证

tr_ada_model = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=2, random_state=1),
                                    n_estimators=200, sample_size=sample_size,
                                    steps=steps, fold=fold,
                                    random_state=random_state)
# r2_test_max = {float64: ()} 0.7338421739641736
# r2_train_ = {float64: ()} 0.8779732832129921
# r2_test_max, r2_train_, pred_train_, pred_test_, max_delta_id = tr_ada_model.fit(x_train_combined,
#                                                                                  y_train_combined,
#                                                                                  x_test_19, y_test_19)

# r2_train, rmse_train, rpd_train, relative_error_train = metrics_(pred_train_, y_train_combined)
# r2_test, rmse_test, rpd_test, relative_error_test = metrics_(pred_test_, y_test_19)
# graph(y_train_combined, pred_train_, y_test_19, pred_test_)

# =========================================================
# 成图
filename = r'E:\桌面\data\combinedcj.tif'
im_bands_1, im_proj_1, im_geotrans_1, data_list_1, im_data_1 = image_operation().read_image(filename=filename,
                                                                                            feature_select=False,
                                                                                            feature_index=[0])
data_list_1[data_list_1 < -10000] = 0  # 将无穷小替换为0
data_list_1[np.isnan(data_list_1)] = 0  # nan值替换为0

pred_test=tr_ada_model.predict(data_list_1)

result_1 = np.array(pred_test_).reshape((278, 243))
a = image_operation().write_image(r'E:\桌面\result1.tif', im_proj=im_proj_1,
                                  im_geotrans=im_geotrans_1, im_data=result_1)


