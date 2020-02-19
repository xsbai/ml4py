import numpy as np


class LinearRegression(object):
    def __init__(self, X, y):
        self.X = np.mat(X)
        self.y = np.mat(y)
        self.number = len(X)
        self.length = len(X[0])
        self.theta = np.ones((self.length, 1))
        self.bias = 0.1
        self.train()

    def cost_function(self, X, y):
        result = np.dot(X, self.theta)
        minus_val = y - result
        cost = 0.5 * sum(np.array(minus_val) ** 2)
        if np.isnan(cost):
            return np.inf
        return cost

    def gradientdescent(self, alpha):
        result = np.dot(self.X, self.theta) + self.bias
        # print(result)
        fro_part = result - self.y
        all_gradient = np.dot(self.X.transpose(), fro_part)
        # print(all_gradient)
        self.theta = self.theta - alpha * all_gradient
        self.bias = self.bias - alpha * fro_part

    def train(self, iterations=10000, alpha=0.01, method=0):
        for k in range(iterations):
            if method == 0:
                self.gradientdescent(alpha)

    def predict(self, X_test):
        # print(np.dot(X_test,self.theta))
        return np.dot(X_test, self.theta)


X = [[1.50], [2.00], [2.50], [3.00], [3.50], [4.00], [6.00]]
y = [[6.450], [7.450], [8.450], [9.450], [11.450], [15.450], [18.450]]
X_test = [[2.5], [6.0]]
y_test = [[8.450], [18.450]]
y = LinearRegression(X, y)
y_test1 = y.predict(X_test)
cost = y.cost_function(X_test, y_test)
print(y_test1)
print(cost)
