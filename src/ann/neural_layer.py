"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
class NeuralLayer:
    def __init__(self, input_size, output_size, init_method='xavier'):
        self.input_size = input_size
        self.output_size = output_size
        self.init_method = init_method
        self.W = self._initialize_weights()
        self.b = np.zeros((output_size, 1))

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
    
    def _initialize_weights(self):
        if self.init_method == 'xavier':
            return np.random.randn(self.output_size, self.input_size) * np.sqrt(2 / (self.input_size + self.output_size))
        elif self.init_method == 'random':
            return np.random.randn(self.output_size, self.input_size) * 0.01
        else:
            raise ValueError('Invalid initialization method')
    
    def forward(self, x):
        self.x = x
        self.z = self.W @ x + self.b
        return self.z
    
    def backward(self, grad_output):
        self.grad_W[:] = np.sum(grad_output @ self.x.transpose(0, 2, 1), axis=0)
        self.grad_b[:] = np.sum(grad_output, axis=0)
        return self.W.T @ grad_output


