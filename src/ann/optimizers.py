"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def step(self, weights, gradients):
        gradients = gradients + self.weight_decay * weights
        weights -= self.learning_rate * gradients


class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = None
    
    def step(self, weights, gradients):
        if self.v is None:
            self.v = np.zeros_like(weights)
        self.v = self.momentum * self.v + self.learning_rate * gradients
        weights -= self.v
        weights -= self.weight_decay * weights*self.learning_rate

class NAG:
    '''
    For this optimizer, compute lookahead weights, get the new gradients and then apply step on the original weights using the new gradients
    '''
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = None

    def step(self, weights, new_gradients):
        if self.v is None:
            self.v = np.zeros_like(weights)
        
        self.v = self.momentum * self.v + self.learning_rate * new_gradients
        weights -= self.v
        weights -= self.weight_decay * weights*self.learning_rate

    def compute_lookahead(self, copy_weights, gradients):
            if self.v is None:
                self.v = np.zeros_like(copy_weights)
            copy_weights = copy_weights - self.momentum * self.v
    

class RMSProp:
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.squared = None
    
    def step(self, weights, gradients):
        if self.squared is None:
            self.squared = np.zeros_like(weights)
        self.squared = self.beta * self.squared + (1 - self.beta) * np.square(gradients)
        weights -= self.learning_rate * gradients / (np.sqrt(self.squared) + self.epsilon)
        weights -= self.weight_decay * weights * self.learning_rate

class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
    
    def step(self, weights, gradients):
        if self.m is None:
            self.m = np.zeros_like(weights)
        if self.v is None:
            self.v = np.zeros_like(weights)

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients #momentum
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradients) #RMSProp
        m_hat = self.m / (1 - self.beta1) #corrected momentum
        v_hat = self.v / (1 - self.beta2) #corrected RMSProp
        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon) #Adam
        weights -= self.weight_decay * weights * self.learning_rate

class Nadam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, weights, gradients):
        if self.m is None:
            self.m = np.zeros_like(weights)
        if self.v is None:
            self.v = np.zeros_like(weights)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients #momentum
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradients) #RMSProp
        m_hat = self.m / (1 - self.beta1) #corrected momentum
        v_hat = self.v / (1 - self.beta2) #corrected RMSProp

        m_nestrov = self.beta1 * m_hat + (1 - self.beta1) * (gradients)/(1- self.beta1**self.t)

        weights -= (self.learning_rate * m_nestrov) / (np.sqrt(v_hat) + self.epsilon)
        weights -= self.weight_decay * weights * self.learning_rate

def get_optimiser(optimiser_type, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
    if optimiser_type == 'sgd':
        return SGD(learning_rate, weight_decay=weight_decay)
    elif optimiser_type == 'momentum':
        return Momentum(learning_rate, beta1, weight_decay=weight_decay)
    elif optimiser_type == 'nag':
        return NAG(learning_rate, beta1, weight_decay=weight_decay)
    elif optimiser_type == 'rmsprop':
        return RMSProp(learning_rate, beta1, epsilon, weight_decay=weight_decay)
    elif optimiser_type == 'adam':
        return Adam(learning_rate, beta1, beta2, epsilon, weight_decay=weight_decay)
    elif optimiser_type == 'nadam':
        return Nadam(learning_rate, beta1, beta2, epsilon, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimiser type')