#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *
class KMeans():
    # 加载数据
    def loadDataSet(self, fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
        dataMat = []  # 文件的最后一个字段是类别标签
        fr = open(fileName)
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = map(float, curLine)  # 将每个元素转成float类型
            dataMat.append(fltLine)
        return dataMat


    # 计算欧几里得距离
    def distEclud(self, vecA, vecB):
        return sqrt(sum(power(vecA - vecB, 2)))  # 求两个向量之间的距离


    # 构建聚簇中心，取k个(此例中k=4)随机质心
    def randCent(self, dataSet, k):
        n = shape(dataSet)[1]
        centroids = mat(zeros((k, n)))  # 每个质心有n个坐标值，总共要k个质心
        for j in range(n):
            minJ = min(dataSet[:, j])
            maxJ = max(dataSet[:, j])
            rangeJ = float(maxJ - minJ)
            centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
        return centroids


    # k-means 聚类算法
    def kMeans(self, dataSet, k, distMeans=distEclud, createCent=randCent):
        '''
        :param dataSet:  没有lable的数据集  (本例中是二维数据)
        :param k:  分为几个簇
        :param distMeans:    计算距离的函数
        :param createCent:   获取k个随机质心的函数
        :return: centroids： 最终确定的 k个 质心
                clusterAssment:  该样本属于哪类  及  到该类质心距离
        '''
        m = shape(dataSet)[0]  # m=80,样本数量
        clusterAssment = mat(zeros((m, 2)))
        # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离，
        centroids = createCent(dataSet, k)
        clusterChanged = True  # 用来判断聚类是否已经收敛
        while clusterChanged:
            clusterChanged = False
            for i in range(m):  # 把每一个数据点划分到离它最近的中心点
                minDist = inf
                minIndex = -1
                for j in range(k):
                    distJI = distMeans(centroids[j, :], dataSet[i, :])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True  # 如果分配发生变化，则需要继续迭代
                clusterAssment[i, :] = minIndex, minDist ** 2  # 并将第i个数据点的分配情况存入字典
            # print centroids
            for cent in range(k):  # 重新计算中心点
                ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 去第一列等于cent的所有列
                centroids[cent, :] = mean(ptsInClust, axis=0)  # 算出这些数据的中心点
        return centroids, clusterAssment

if __name__ == '__main__':
    datMat = mat(KMeans().loadDataSet('kmeansdata.txt'))
    # print min(datMat[:,0])
    # print max(datMat[:,1])
    # print randCent(datMat,4)
    myCentroids, clustAssing = KMeans().kMeans(datMat, 4)
    print(myCentroids)
    # print clustAssing,len(clustAssing)
