"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np

class Activation_base:
    def __init__(self):
        pass
    def forward(self, x):
        pass
    
    def backward(self, grad_output):
        pass

class ReLU(Activation_base):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self.x = x
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, grad_output):
        return np.where(self.x > 0, 1, 0) * grad_output

class Sigmoid(Activation_base):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self.x = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, grad_output):
        return self.output * (1 - self.output) * grad_output

class Tanh(Activation_base):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self.x = x
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        return (1 - self.output**2) * grad_output

class Softmax(Activation_base):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self.x = x
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        sum_grad = np.sum(grad_output * self.output, axis=1, keepdims=True)
        return self.output * (grad_output - sum_grad)

def get_activation(activation_type):
    if activation_type == 'relu':
        return ReLU()
    elif activation_type == 'sigmoid':
        return Sigmoid()
    elif activation_type == 'tanh':
        return Tanh()
    elif activation_type == 'softmax':
        return Softmax()
    else:
        raise ValueError('Invalid activation type')
