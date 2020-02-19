#!/usr/bin/env python
# -*- coding: utf-8 -*-

class DimensionValueError(ValueError):
    """定义异常类"""
    pass


class PCA(object):
    """定义PCA类"""
    def __init__(self, x, n_components=None):
        self.x = x
        self.dimension = x.shape[1]
 
        if n_components and n_components >= self.dimension:
            raise DimensionValueError("n_components error")
 
        self.n_components = n_components


    def cov(self):
        """求x的协方差矩阵"""
        x_T = np.transpose(self.x)                           #矩阵转置
        x_cov = np.cov(x_T)                                  #协方差矩阵
        return x_cov
 
    def get_feature(self):
        """求协方差矩阵C的特征值和特征向量"""
        x_cov = self.cov()
        a, b = np.linalg.eig(x_cov)
        m = a.shape[0]
        c = np.hstack((a.reshape((m,1)), b))
        c_df = pd.DataFrame(c)
        c_df_sort = c_df.sort(columns=0, ascending=False)    
        return c_df_sort

    def reduce_dimension(self):
        """指定维度降维和根据方差贡献率自动降维"""
        c_df_sort = self.get_feature()
        varience = self.explained_varience_()
 
        if self.n_components:                                #指定降维维度
            p = c_df_sort.values[0:self.n_components, 1:]
            y = np.dot(p, np.transpose(self.x))              
            return np.transpose(y)
 
        varience_sum = sum(varience)                         
        varience_radio = varience / varience_sum
 
        varience_contribution = 0
        for R in xrange(self.dimension):
            varience_contribution += varience_radio[R]       
            if varience_contribution >= 0.99:
                break
 
        p = c_df_sort.values[0:R+1, 1:]                      #取前R个特征向量
        y = np.dot(p, np.transpose(self.x))                  
        return np.transpose(y)




if __name__ == '__main__':
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target
    pca = PCA(x)
    y = pca.reduce_dimension()