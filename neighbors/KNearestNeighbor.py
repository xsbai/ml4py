# -*- coding: utf-8 -*-

"""kNN最近邻算法最重要的三点：
   (1)确定k值。k值过小，对噪声非常敏感；k值过大，容易误分类
   (2)采用适当的临近性度量。对于不同的类型的数据，应考虑不同的度量方法。除了距离外，也可以考虑相似性。
   (3)数据预处理。需要规范数据，使数据度量范围一致。
"""

import pandas as pd
import numpy as np


class kNN:
    def __init__(self, X, y=None, test='YES'):
        """参数X为训练样本集，支持list，array和DataFrame；

           参数y为类标号，支持list,array,Series
           默认参数y为空值，表示类标号字段没有单独列出来，而是存储在数据集X中的最后一个字段；
           参数y不为空值时，数据集X中不能含有字段y

           参数test默认为'YES'，表是将原训练集拆分为测试集和新的训练集
        """
        if isinstance(X, pd.core.frame.DataFrame) != True:  # 将数据集转换为DataFrame格式
            self.X = pd.DataFrame(X)
        else:
            self.X = X
        if y is None:  # 将特征和类别分开
            self.y = self.X.iloc[:, -1]
            self.X = self.X.iloc[:, :-1]
            self.max_data = np.max(self.X, axis=0)  # 获取每个特征的最大值，为下面规范数据用
            self.min_data = np.min(self.X, axis=0)  # 获取每个特征的最小值，为下面规范数据用
            max_set = np.zeros_like(self.X)
            max_set[:] = self.max_data  # 以每个特征的最大值，构建一个与训练集结构一样的数据集
            min_set = np.zeros_like(self.X)
            min_set[:] = self.min_data  # 以每个特征的最小值，构建一个与训练集结构一样的数据集
            self.X = (self.X - min_set) / (max_set - min_set)  # 规范训练集
        else:
            self.max_data = np.max(self.X, axis=0)
            self.min_data = np.min(self.X, axis=0)
            max_set = np.zeros_like(self.X)
            max_set[:] = self.max_data
            min_set = np.zeros_like(self.X)
            min_set[:] = self.min_data
            self.X = (self.X - min_set) / (max_set - min_set)
            if isinstance(y, pd.core.series.Series) != True:
                self.y = pd.Series(y)
            else:
                self.y = y
        if test == 'YES':  # 如果test为'YES'，将原训练集拆分为测试集和新的训练集
            self.test = 'YES'  # 设置self.test，后面knn函数判断测试数据需不需要再规范
            allCount = len(self.X)
            dataSet = [i for i in range(allCount)]
            testSet = []
            for i in range(int(allCount * (1 / 5))):
                randomnum = dataSet[int(np.random.uniform(0, len(dataSet)))]
                testSet.append(randomnum)
                dataSet.remove(randomnum)
            self.X, self.testSet_X = self.X.iloc[dataSet], self.X.iloc[testSet]
            self.y, self.testSet_y = self.y.iloc[dataSet], self.y.iloc[testSet]
        else:
            self.test = 'NO'

    def getDistances(self, point):  # 计算训练集每个点与计算点的欧几米得距离
        points = np.zeros_like(self.X)  # 获得与训练集X一样结构的0集
        points[:] = point
        minusSquare = (self.X - points) ** 2
        EuclideanDistances = np.sqrt(minusSquare.sum(axis=1))  # 训练集每个点与特殊点的欧几米得距离
        return EuclideanDistances

    def getClass(self, point, k):  # 根据距离最近的k个点判断计算点所属类别
        distances = self.getDistances(point)
        argsort = distances.argsort(axis=0)  # 根据数值大小，进行索引排序
        classList = list(self.y.iloc[argsort[0:k]])
        classCount = {}
        for i in classList:
            if i not in classCount:
                classCount[i] = 1
            else:
                classCount[i] += 1
        maxCount = 0
        maxkey = 'x'
        for key in classCount.keys():
            if classCount[key] > maxCount:
                maxCount = classCount[key]
                maxkey = key
        return maxkey

    def knn(self, testData, k):  # kNN计算，返回测试集的类别
        if self.test == 'NO':  # 如果self.test == 'NO'，需要规范测试数据（参照上面__init__）
            testData = pd.DataFrame(testData)
            max_set = np.zeros_like(testData)
            max_set[:] = self.max_data
            min_set = np.zeros_like(testData)
            min_set[:] = self.min_data
            testData = (testData - min_set) / (max_set - min_set)  # 规范测试集
        if testData.shape == (len(testData), 1):  # 判断testData是否是一行记录
            label = self.getClass(testData.iloc[0], k)
            return label  # 一行记录直接返回类型
        else:
            labels = []
            for i in range(len(testData)):
                point = testData.iloc[i, :]
                label = self.getClass(point, k)
                labels.append(label)
            return labels  # 多行记录则返回类型的列表

    def errorRate(self, knn_class, real_class):  # 计算kNN错误率,knn_class为算法计算的类别，real_class为真实的类别
        error = 0
        allCount = len(real_class)
        real_class = list(real_class)
        for i in range(allCount):
            if knn_class[i] != real_class[i]:
                error += 1
        return error / allCount
