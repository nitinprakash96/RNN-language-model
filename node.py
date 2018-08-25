import numpy as np


class MultiplyNode:
    def forward_pass(self, W, x):
        output = np.dot(W, x)
        return output

    def backward_pass(self, W, x, dz):
        # Think of a more readable way
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dx = np.dot(np.transpose(W), dz)
        return dW, dx


class SumNode:
    def forward_pass(self, x, b):
        output = x + b
        return output

    def backward_pass(self, x, y, dz):
        dx = dz * np.ones_like(x)
        dy = dz * np.ones_like(y)
        return dx, dy