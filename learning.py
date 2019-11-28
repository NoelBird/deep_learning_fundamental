import sys
import os
import numpy as np
from common import softmax, error_cross_entropy
from common import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = error_cross_entropy(y, t)
        return loss

