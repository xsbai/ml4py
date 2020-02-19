import numpy as np
import matplotlib.pyplot as plt


class LogisticsRegression(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.b = 0.1
        self.length = len(X[0])
        self.num = len(X)
        self.theta = np.ones((self.length, 1))

    def sigmoid(self, re):
        return 1.0 / (1.0 + np.exp(re))

    def cost_function(self):
        cost_sum = 0.0
        for i in range(self.num):
            cost_tem = self.y[i] * np.exp(self.sigmoid(np.dot(self.theta, np.array(self.X[i])) + self.b)) + \
                       (1 - self.y[i]) * np.exp(1 - self.sigmoid(np.dot(self.theta, np.array(self.X[i])) + self.b))
            cost_sum += cost_tem
        return -1 * cost_sum / self.num

    def gradientdescent(self, alpha):
        mat_X = np.mat(self.X)
        mat_y = np.mat(self.y)
        result = self.sigmoid(np.dot(mat_X, self.theta) + self.b)
        # print(result)
        fro_part = result - mat_y
        all_gradient = np.dot(mat_X.transpose(), fro_part)
        # print(all_gradient)
        self.theta = self.theta - alpha * all_gradient
        self.b = self.b - alpha * fro_part

    def stoc_grad_ascent_one(self, alpha):  # 随机梯度下降
        dataIndex = list(range(self.num))
        for i in range(self.num):
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = self.sigmoid(np.dot(self.theta.transpose(), np.array(self.X[i])) + self.b)  # 数值计算
            error = h - self.y[i]
            split_x = [[j] for j in self.X[i]]
            self.theta = self.theta - alpha * error * np.array(split_x)
            self.b = self.b - alpha * error
            del (dataIndex[randIndex])

    def train(self, iterations=100, alpha=0.1, method=1):
        for k in range(iterations):
            if method == 0:
                self.gradientdescent(alpha)
            if method == 1:
                self.stoc_grad_ascent_one(alpha)
        return self.theta, self.b

    def predict(self, X_test):
        val = self.sigmoid(np.dot(self.theta.transpose(), np.array(X_test)) + self.b)
        return 1 if val > 0.5 else 0


X = [[-0.017612, 14.053064], [-1.395634, 4.662541], [-0.752157, 6.538620], [-1.322371, 7.152853], [0.423363, 11.054677],
     [0.406704, 7.067335], [0.667394, 12.741452], [-2.460150, 6.866805], [0.569411, 9.548755], [-0.026632, 10.427743]]

y = [[0], [1], [0], [0], [0], [1], [0], [1], [0], [0]]

lr = LogisticsRegression(X, y)
theat, b = lr.train()
print(theat)
print(b)
