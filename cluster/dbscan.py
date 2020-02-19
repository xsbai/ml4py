#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import random
class DBScan:
    def  dist(a, b):
        # 计算a,b两个元组的欧几里得距离
        return math.sqrt(np.power(a-b, 2).sum())

    def my_dbscanl(dataSet, eps, minPts):
        # numpy.ndarray的 shape属性表示矩阵的行数与列数
        nPoints = dataSet.shape[0]
        # (1)标记所有对象为unvisited
        # 在这里用一个类vPoints进行买现
        vPoints = visitlist(count=nPoints)
        # 初始化簇标记列表C,簇标记为 k
        k = -1
        C = [-1 for i in range(nPoints)]
        while(vPoints.unvisitednum > 0):
            # (3)随机上选择一个unvisited对象p
            P = random.choice(vPoints.unvisitedlist)
            # (4)标记p为visited
            vPoints.visit(p)
            # (5)if p的$\varepsilon$-邻域至少有MinPts个对象
            # N是p的$\varepsilon$-邻域点列表
            N = [i for i in range(nPoints) if dist(dataSet[i], dataSet[p])<= eps]
            if  len(N) >= minPts:
                # (6)创建个新簇C，并把p添加到C
                # 这里的C是一个标记列表，直接对第p个结点进行赋植
                k += 1
                C[p]=k
                # (7)令N为p的ε-邻域中的对象的集合
                # N是p的$\varepsilon$-邻域点集合
                # (8) for N中的每个点p'
                for p1 in N:
                    # (9) if p'是unvisited
                    if p1 in vPoints.unvisitedlist:
                        # (10)标记p’为visited
                        vPoints.visit(p1)
                        # (11) if p'的$\varepsilon$-邻域至少有MinPts个点，把这些点添加到N
                        # 找出p'的$\varepsilon$-邻域点，并将这些点去重添加到N
                        M=[i for i in range(nPoints) if dist(dataSet[i], \
                            dataSet[p1]) <= eps]
                        if len(M) >= minPts:
                            for i in M:
                                if i not in N:
                                    N.append(i)
                        # (12) if p'还不是任何簇的成员，把P'添加到C
                        # C是标记列表，直接把p'分到对应的簇里即可
                        if  C[p1] == -1:
                            C[p1]= k
            # (15)else标记p为噪声
            else:
                C[p]=-1

        # (16)until没有标t己为unvisitedl内对象
        return C


    from scipy.spatial import KDTree

    def my-dbscan2(dataSet, eps, minPts):
        # numpy.ndarray的 shape属性表示矩阵的行数与列数
        # 行数即表小所有点的个数
        nPoints = dataSet.shape[0]
        # (1) 标记所有对象为unvisited
        # 在这里用一个类vPoints进行实现
        vPoints = visitlist(count=nPoints)
        # 初始化簇标记列表C，簇标记为 k
        k = -1
        C = [-1 for i in range(nPoints)]
        # 构建KD-Tree，并生成所有距离<=eps的点集合
        kd = KDTree(X)
        while(vPoints.unvisitednum>0):
            # (3) 随机选择一个unvisited对象p
            p = random.choice(vPoints.unvisitedlist)
            # (4) 标t己p为visited
            vPoints.visit(p)
            # (5) if p 的$\varepsilon$-邻域至少有MinPts个对象
            # N是p的$\varepsilon$-邻域点列表
            N = kd.query_ball_point(dataSet[p], eps)
            if len(N) >= minPts:
                # (6) 创建个一个新簇C，并把p添加到C
                # 这里的C是一个标记列表，直接对第p个结点进行赋值
                k += 1
                C[p] = k
                # (7) 令N为p的$\varepsilon$-邻域中的对象的集合
                # N是p的$\varepsilon$-邻域点集合
                # (8) for N中的每个点p'
                for p1 in N:
                    # (9) if p'是unvisited
                    if p1 in vPoints.unvisitedlist:
                        # (10) 标记p'为visited
                        vPoints.visit(p1)
                        # (11) if p'的$\varepsilon$-邻域至少有MinPts个点，把这些点添加到N
                        # 找出p'的$\varepsilon$-邻域点，并将这些点去重新添加到N
                        M = kd.query_ball_point(dataSet[p1], eps)
                        if len(M) >= minPts:
                            for i in M:
                                if i not in N:
                                    N.append(i)
                        # (12) if p'还不是任何簇的成员，把p'添加到c
                        # C是标记列表，直接把p'分到对应的簇里即可
                        if C[p1] == -1
                            C[p1] = k
            # (15) else标记p为噪声
            else:
                C[p1] = -1

        # (16) until没有标记为unvisited的对象
        return C