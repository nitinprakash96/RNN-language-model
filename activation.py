import numpy as np


class tanh:
    """
    This activation squashes the output in the range -1 and 1.
    """

    def forward_pass(self, bottom):
        output = np.tanh(bottom)
        return output

    def backward_pass(self, top):
        x = self.forward_pass(top)
        output = (1.0 - np.square(x))
        return


class sigmoid:
    """
    This activation squashes the output in the range 0 and 1.
    """

    def forward_pass(self, bottom):
        output = 1.0 / (1.0 * np.exp(-bottom))
        return output

    def backward_pass(self, top, top_diff):
        x = self.forward_pass(top)
        output = x * (1.0 - x) * top_diff
        return output


# source: https://stackoverflow.com/a/47936476/6244324
class relu:
    """
    This activation gives an output x if x is positive and 0 otherwise.
    Mathematically this can be represented as: A(x) = max(0,x)
    """

    def forward_pass(self, bottom):
        output = bottom * (bottom > 0)
        return output

    def backward_pass(self, top):
        x = self.forward_pass(top)
        output = 1.0 * (x > 0)
        return output