"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

try:
    from .activations import Softmax
except Exception:
    try:
        from ann.activations import Softmax
    except Exception:
        from da6401_assignment_1.src.ann.activations import Softmax

class CrossEntropyLoss:
    def __init__(self):
        self.y_pred = None
        self.softmax = Softmax()
    
    def forward(self, y_true, y_pred_logits):
        # Apply softmax to logits
        y_pred = self.softmax.forward(y_pred_logits)
        self.y_pred = y_pred
        return -np.mean(y_true * np.log(y_pred + 1e-12))
    
    def backward(self, y_true, y_pred_logits):
        # Apply softmax to logits
        y_pred = self.softmax.forward(y_pred_logits)
        self.y_pred = y_pred
        goutput = (-y_true / (y_pred + 1e-12)) / y_true.size
        return self.softmax.backward(goutput)

class MSELoss:
    def __init__(self):
        pass
    
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true, y_pred):
        return (-2 * (y_true - y_pred)) / y_true.size
