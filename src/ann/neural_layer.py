"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class NeuralLayer:
    def __init__(self, input_size, output_size, init_method="xavier"):
        self.input_size = input_size
        self.output_size = output_size
        self.init_method = init_method
        self.W = self._initialize_weights()
        self.b = np.zeros((output_size, 1), dtype=np.float32)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def _initialize_weights(self):
        if self.init_method == "xavier":
            return (np.random.randn(self.output_size, self.input_size).astype(np.float32)* np.sqrt(2.0 / (self.input_size + self.output_size)))
        elif self.init_method == "random":
            return (np.random.randn(self.output_size, self.input_size).astype(np.float32) * 0.01)
        else:
            raise ValueError("Invalid initialization method")

    def forward(self, x):
        self.input_ndim = x.ndim

        if x.ndim == 3:
            self.x = x
            self.z = self.W @ x + self.b
            return self.z

        if x.ndim == 2:
            if x.shape[1] == self.input_size:
                self.x = x
                self.z = x @ self.W.T + self.b.T
                return self.z
            if x.shape == (self.input_size, 1):
                self.x = x.reshape(1, self.input_size, 1)
                self.z = self.W @ self.x + self.b
                return self.z.reshape(1, self.output_size)

        raise ValueError(f"Unexpected input shape for layer: {x.shape}")

    def backward(self, grad_output):
        if self.input_ndim == 3:
            self.grad_W[:] = np.sum(grad_output @ self.x.transpose(0, 2, 1), axis=0)
            self.grad_b[:] = np.sum(grad_output, axis=0)
            return self.W.T @ grad_output

        if self.input_ndim == 2:
            if grad_output.ndim == 3:
                grad_output = grad_output.reshape(grad_output.shape[0], grad_output.shape[1])
            self.grad_W[:] = grad_output.T @ self.x
            self.grad_b[:] = np.sum(grad_output, axis=0).reshape(self.output_size, 1)
            return grad_output @ self.W

        raise ValueError(f"Unexpected gradient shape for layer: {grad_output.shape}")


