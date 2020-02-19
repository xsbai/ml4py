import numpy as np


class Perceptron(object):
    def __init__(self, X, y):
        self.X = np.array(X)
        self.length = len(X[0])
        self.y = np.array(y)
        self.w = np.ones((self.length))
        self.b = 0

    def process(self, delta, iterations=100):
        for i in range(iterations):
            choice = -1
            for j in range(len(self.X)):
                if np.sign(np.dot(self.w, self.X[j]) + self.b) != self.y[j]:
                    choice = j
                    break
            if choice == -1:
                break
            self.w = self.w + delta * self.y[choice] * self.X[choice]
            self.b = self.b + delta * self.y[choice]

    def predict(self, x_test):
        x_test = np.array(x_test)
        return np.sign(np.dot(self.w, x_test) + self.b)


x_train = [[0.5, 0], [0, 0.5], [1, 0.5], [0.5, 1]]
y_train = [-1, -1, 1, 1]
x_test = [1, 1]
perceptron = Perceptron(x_train, y_train)
perceptron.process(1, 100)
y = perceptron.predict(x_test)
print(y)
