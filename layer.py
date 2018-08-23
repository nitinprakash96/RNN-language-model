import numpy as np

from activation import Tanh, Sigmoid, Relu
from node import MultiplyNode, SumNode

mul = MultiplyNode()
add = SumNode()
activation = Tanh()


class Layer:
    def forward(self, x, U, W, V, prev):
        self.mulU = mul.forward_pass(U, x)
        self.mulW = mul.forward_pass(W, prev)
        self.add = add.forward_pass(self.mulW, self.mulU)
        self.act = activation.forward_pass(self.add)
        self.mulV = mul.forward_pass(V, self.act)

    def backward(self, x, U, W, V, prev, diff_act, dmulV):
        self.forward(x, U, W, V, prev)
        dV, dactV = mul.backward_pass(V, self.act, dmulV)
        dact = dactV + diff_act
        dsum = activation.backward_pass(self.add, dact)
        dmulW, dmulU = add.backward_pass(self.mulW, self.mulU, dsum)
        dW, dprev = mul.backward_pass(W, prev, dmulW)
        dU, dx = mul.backward_pass(U, x, dmulU)
        return (dU, dW, dV, dprev)
