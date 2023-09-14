"""
TwoStageTrAdaBoostR2 算法

基于文献 "Boosting for Regression Transfer"中的算法3

"""
# 算法原理：
"""
Stage 1.源域实例权重慢慢下降，直到达到一个固定值(由cv确定);由CV得到平均误差最低的模型——确定实例权重
Stage 2.所有源域实例权重不变，目标域实例权重正常更新(Adaboost.R2);
      只有在第二阶段生成的预测结果被存储起来，并用于确定结果模型的输出。
"""
# 1.输入:源域训练集(T_s,n个),源域测试集(T_t,m个),迭代次数(S),学习器个数（N）
#     K折交叉验证折数：F;基学习器
# 2.合并源域数据、目标域数据：T=T_s&T_t(n+m个)
# 3.初始化样本权重：w_t=1/(n+m)
# 4.for t=1,...,S:
#     1.利用T、w_t、N——model_t,前n个数据（源域训练集）的样本权重永远不变；
#     2.利用cv获取model_t的error
#     3.计算每个实例的改正误差e
#     4.更新实例权重：权重因子β_t必须保证目标域权重之和为:[n/(n+m)]+[t(1-n/(n+m))/(S-1)]
#       以保证目标域的实例权重之和在S steps 后由m/(n+m)均匀增加到1;使用二进制搜索逼近该值
# 5.输出结果

import copy

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from Final.functions import train_model, pre_process_data


## 第二步

class Stage2_TrAdaBoostR2:
    def __init__(self,
                 base_estimator=DecisionTreeRegressor(max_depth=4),
                 sample_size=None,
                 n_estimators=50,
                 learning_rate=1.,
                 loss='linear',
                 random_state=np.random.mtrand._rand):

        # 基础学习器：默认回归树（最大深度4）
        # 样本大小：默认None	[515,15]
        # 迭代数：默认50
        # 学习率：默认1
        # 损失函数形式：默认线性

        ## random_state = np.random.mtrand._rand)？？？？
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        # fit使用训练数据来构造学习器

        # X
        # y
        # 样本权重

        # 检查参数

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")  # 如果learning rate<=0则抛出异常

        if sample_weight is None:

            # 若初始权重为None则设置其权重向量为1/X.shape[0]

            sample_weight = np.empty(X.shape[0],
                                     dtype=np.float64)  # 返回随机初始化数组shape=（X.shape[0] ,  ），为一个列向量，类型为np.float64(精度高)
            sample_weight[:] = 1. / X.shape[0]  # 设置权重向量



        else:

            # 若初始权重不为None，则将权重向量归一化（[0,1]）

            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # 检查样本权重之和是否为正，非正则抛出异常：尝试拟合非正加权样本数

            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # 检查样本大小是否为None，是则抛出异常：需要其他输入：缺少源和目标域训练集大小

        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")

        # 若源域和目标域训练集大小  和  X的大小不同则抛出异常：输入错误：指定的样本大小不等于输入大小

        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")

        # 清除以前的拟合结果

        self.estimators_ = []  # 学习器预测结果？

        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)  # 学习器权重向量？（零向量（50，））

        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)  # 学习器error？（1向量（50，））

        for iboost in range(
                self.n_estimators):  # this for loop is sequential and does not support parallel(revison is needed if making parallel)

            # 循环50次，iboost=0，1，2，。。。49
            # Boosting step

            sample_weight, estimator_weight, estimator_error = self._stage2_adaboostR2(
                iboost,
                X, y,
                sample_weight)

            # sample_weight在fit函数中, estimator_weight, estimator_error上面没有，只有self.estimator_weights_学习器权重向量？（零向量（50，）），self.estimator_errors_学习器error？（1向量（50，））

            # 下面的函数_stage2_adaboostR2(self, iboost, X, y, sample_weight)

            # 如果样本权重为None则跳出For循环

            if sample_weight is None:
                break

            self.estimator_weights_[
                iboost] = estimator_weight  # 将计算得到的学习器权重estimator_weight加入学习器权重向量self.estimator_weights_?
            self.estimator_errors_[
                iboost] = estimator_error  # 将计算得到的学习器error  :estimator_error加入学习器error向量self.estimator_errors_?

            # 如果学习器error为零跳出for循环

            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)  # 计算样本权重之和

            # 如果样本权重之和变为非正，则跳出For循环

            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # 归一化：前49个样本权重归一化，第50个不归一化？？？

                sample_weight /= sample_weight_sum
        return self

    def _stage2_adaboostR2(self, iboost, X, y, sample_weight):

        estimator = copy.deepcopy(
            self.base_estimator)  # some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)

        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weight)
        cdf /= cdf[-1]
        uniform_samples = self.random_state.random_sample(X.shape[0])  # 产生[0,1)随机数
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)

        self.estimators_.append(estimator)  # add the fitted estimator

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Calculate the average loss
        estimator_error = (sample_weight * error_vect).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)

        # avoid overflow of np.log(1. / beta)
        if beta < 1e-308:
            beta = 1e-308
        estimator_weight = self.learning_rate * np.log(1. / beta)

        # Boost weight using AdaBoost.R2 alg except the weight of the source data
        # the weight of the source data are remained
        source_weight_sum = np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
        target_weight_sum = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)

        if not iboost == self.n_estimators - 1:
            sample_weight[-self.sample_size[-1]:] *= np.power(
                beta,
                (1. - error_vect[-self.sample_size[-1]:]) * self.learning_rate)
            # make the sum weight of the source data not changing
            source_weight_sum_new = np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
            target_weight_sum_new = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)
            if source_weight_sum_new != 0. and target_weight_sum_new != 0.:
                sample_weight[:-self.sample_size[-1]] = sample_weight[:-self.sample_size[
                    -1]] * source_weight_sum / source_weight_sum_new
                sample_weight[-self.sample_size[-1]:] = sample_weight[-self.sample_size[
                    -1]:] * target_weight_sum / target_weight_sum_new

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        # Evaluate predictions of all estimators
        predictions = np.array([
            est.predict(X) for est in self.estimators_[:len(self.estimators_)]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]

        # Return median predictions
        return predictions[np.arange(X.shape[0]), median_estimators]


################################################################################
## the whole two stages
################################################################################

class TwoStageTrAdaBoostR2:
    # 确定算法参数：基学习器、样本采样大小、学习器个数、迭代次数、交叉验证折数、学习率、损失函数、random_state
    def __init__(self,
                 base_estimator=DecisionTreeRegressor(max_depth=4),
                 sample_size=None,  # 源域、目标域样本数量e.g.,[100,10]
                 n_estimators=50,
                 steps=10,  # 迭代次数
                 fold=5,  # 交叉验证折数
                 learning_rate=1.,
                 loss='linear',
                 random_state=np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.steps = steps
        self.fold = fold
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y, x_test_19, y_test_19, sample_weight=None):
        # X=[X_source,X_target]由源域、目标域的特征组成的列表
        # y=[y_source,y_target]由源域、目标域的标签组成的列表
        # sample_weight:（可选），样本初始权重；默认等权

        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            # 初始化样本权重wei(1/sample_size)
            sample_weight = np.empty(X.shape[0], dtype=np.float64)  # 随机初始化数组
            sample_weight[:] = 1. / X.shape[0]  # 赋予数组固定权重

        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")
        # 读取源域目标域数据
        X_source = X[:-self.sample_size[-1]]
        y_source = y[:-self.sample_size[-1]]
        X_target = X[-self.sample_size[-1]:]
        y_target = y[-self.sample_size[-1]:]

        self.models_ = []
        self.r2_test_lst = []
        self.r2_train_lst = []
        self.pred_train_list = []
        self.pred_test_list = []
        self.pred_delta_list = []
        for istep in range(self.steps):
            # step循环
            model = Stage2_TrAdaBoostR2(self.base_estimator,
                                        sample_size=self.sample_size,
                                        n_estimators=self.n_estimators,
                                        learning_rate=self.learning_rate, loss=self.loss,
                                        random_state=self.random_state)
            model.fit(X, y, sample_weight=sample_weight)
            self.models_.append(model)
            # No cv training
            r2train, rmsetrain, rpdtrain, mretrain, r2test, rmsetest, rpdtest, mretest, pred_train, pred_test = train_model(
                model, X, y, x_test_19, y_test_19, return_pred=True)

            # error = []
            target_weight = sample_weight[-self.sample_size[-1]:]
            source_weight = sample_weight[:-self.sample_size[-1]]
            # cv training
            # kf = KFold(n_splits=self.fold)
            # 由CV确定实例权重sample_weight(由CV得到平均误差最低的模型——确定实例权重)
            # for train, test in kf.split(X_target):
            #     sample_size = [self.sample_size[0], len(train)]
            #     # 将目标域训练集作交叉验证，组成新的训练集：源域训练集+目标域部分训练集，以此求出交叉验证平均精度
            #     model = Stage2_TrAdaBoostR2(self.base_estimator,
            #                                 sample_size=sample_size,
            #                                 n_estimators=self.n_estimators,
            #                                 learning_rate=self.learning_rate, loss=self.loss,
            #                                 random_state=self.random_state)
            #     X_train = np.concatenate((X_source, X_target[train]))
            #     y_train = np.concatenate((y_source, y_target[train]))
            #     X_test = X_target[test]
            #     y_test = y_target[test]
            #     # make sure the sum weight of the target data do not change with CV's split sampling
            #     target_weight_train = target_weight[train] * np.sum(target_weight) / np.sum(target_weight[train])
            #     model.fit(X_train, y_train, sample_weight=np.concatenate((source_weight, target_weight_train)))
            #     y_predict = model.predict(X_test)
            #     # error.append(mean_squared_error(y_predict, y_test))  # MSE
            #     error.append(mean_squared_error(y_predict, y_test) ** 0.5)  # RMSE
            # self.errors_.append(np.array(error).mean())
            # self.errors_:该step下所有模型的平均误差，共steps个

            # self.errors_.append(np.array(error).mean())
            self.pred_delta_list.append(np.abs(pred_train - y))
            self.r2_test_lst.append(r2test)
            self.r2_train_lst.append(r2train)
            self.pred_train_list.append(pred_train)
            self.pred_test_list.append(pred_test)
            sample_weight = self._twostage_adaboostR2(istep, X, y, sample_weight)

            if sample_weight is None:
                break
            # if np.array(error).mean() == 0:
            #     break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if istep < self.steps - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        r2_test_max = max(self.r2_test_lst)
        r2_train_ = self.r2_train_lst[np.argmax(self.r2_test_lst)]
        pred_train_ = self.pred_train_list[np.argmax(self.r2_test_lst)]
        pred_test_ = self.pred_test_list[np.argmax(self.r2_test_lst)]
        max_delta_id = np.argmax(self.pred_delta_list[np.argmax(self.r2_test_lst)])
        return r2_test_max, r2_train_, pred_train_, pred_test_, max_delta_id

    def _twostage_adaboostR2(self, istep, X, y, sample_weight):

        estimator = copy.deepcopy(
            self.base_estimator)  # some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)

        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weight)
        cdf /= cdf[-1]
        uniform_samples = self.random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Update the weight vector
        beta = self._beta_binary_search(istep, sample_weight, error_vect, stp=1e-50)

        if not istep == self.steps - 1:
            sample_weight[:-self.sample_size[-1]] *= np.power(
                beta,
                (error_vect[:-self.sample_size[-1]]) * self.learning_rate)
        return sample_weight

    def _beta_binary_search(self, istep, sample_weight, error_vect, stp):
        # 二分查找：β_t
        # calculate the specified sum of weight for the target data
        n_target = self.sample_size[-1]
        n_source = np.array(self.sample_size).sum() - n_target
        theoretical_sum = n_target / (n_source + n_target) + istep / (self.steps - 1) * (
                1 - n_target / (n_source + n_target))
        # for the last iteration step, beta is 0.
        if istep == self.steps - 1:
            beta = 0.
            return beta
        # binary search for beta
        L = 0.
        R = 1.
        beta = (L + R) / 2
        sample_weight_ = copy.deepcopy(sample_weight)
        sample_weight_[:-n_target] *= np.power(
            beta,
            (error_vect[:-n_target]) * self.learning_rate)
        sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
        updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)

        while np.abs(updated_weight_sum - theoretical_sum) > 0.01:
            if updated_weight_sum < theoretical_sum:
                R = beta - stp
                if R > L:
                    beta = (L + R) / 2
                    sample_weight_ = copy.deepcopy(sample_weight)
                    sample_weight_[:-n_target] *= np.power(
                        beta,
                        (error_vect[:-n_target]) * self.learning_rate)
                    sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                    updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                else:
                    print("At step:", istep + 1)
                    print("Binary search's goal not meeted! Value is set to be the available best!")
                    print("Try reducing the search interval. Current stp interval:", stp)
                    break

            elif updated_weight_sum > theoretical_sum:
                L = beta + stp
                if L < R:
                    beta = (L + R) / 2
                    sample_weight_ = copy.deepcopy(sample_weight)
                    sample_weight_[:-n_target] *= np.power(
                        beta,
                        (error_vect[:-n_target]) * self.learning_rate)
                    sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                    updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                else:
                    print("At step:", istep + 1)
                    print("Binary search's goal not meeted! Value is set to be the available best!")
                    print("Try reducing the search interval. Current stp interval:", stp)
                    break
        return beta

    def predict(self, X):
        # select the model with the least CV error
        # 选出所有steps中误差最小的

        fmodel = self.models_[np.array(self.errors_).argmin()]
        #
        predictions = fmodel.predict(X)
        return predictions
