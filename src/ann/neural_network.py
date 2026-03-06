"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
from ann.optimizers import NAG, get_optimiser
from utils.data_loader import Dataloader

from .neural_layer import NeuralLayer
from .activations import get_activation
import numpy as np
from .objective_functions import CrossEntropyLoss, MSELoss

def _cfg(cli_args):
    return cli_args if isinstance(cli_args, dict) else vars(cli_args)

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """ 
        cli_args = _cfg(cli_args)

        hidden_size = cli_args.get("hidden_size", cli_args.get("hidden_layers", [64]))
        self.input_size = cli_args.get("input_size", 784)
        self.output_size = cli_args.get("output_size", 10)
        self.init_method = cli_args.get("weight_init", cli_args.get("init_method", "xavier"))
        self.activation_type = cli_args.get("activation", "relu")
        self.loss_fn = cli_args.get("loss", cli_args.get("loss_fn", "cross_entropy"))
        self.dataset_name = cli_args.get("dataset", "mnist")

        opt_cfg = cli_args.get("optimizer", "sgd")
        if isinstance(opt_cfg, str):
            self.optimizer = get_optimiser(
                opt_cfg,
                learning_rate=cli_args.get("learning_rate", 0.01),
                weight_decay=cli_args.get("weight_decay", 0.0),
            )
        else:
            self.optimizer = opt_cfg

        self.layers = []
        self.full_layers = []
        total_params = 0
        num_layers = len(hidden_size)
        for i in range(1, len(hidden_size)):
            total_params += hidden_size[i] * hidden_size[i-1] + hidden_size[i]
        total_params += self.output_size * hidden_size[num_layers-1] + self.output_size + self.input_size * hidden_size[0] + hidden_size[0]
        print(f"Total parameters: {total_params}")
        self.global_weights = np.zeros((total_params))
        self.global_grad = np.zeros((total_params))

        if self.loss_fn == 'cross_entropy':
            self.loss_fn = CrossEntropyLoss()
        elif self.loss_fn == 'mse':
            self.loss_fn = MSELoss()
        else:
            raise ValueError('Invalid loss function')

        self.layers = []
        self.full_layers = []

        prev_dim = self.input_size
        for h in hidden_size:
            dense = NeuralLayer(prev_dim, h, self.init_method)
            self.full_layers.append(dense)
            self.layers.append(dense)
            self.full_layers.append(get_activation(self.activation_type))
            prev_dim = h

        out_dense = NeuralLayer(prev_dim, self.output_size, self.init_method)
        self.full_layers.append(out_dense)
        self.layers.append(out_dense)

        self._assign_global_params()
        self.past_input = None
        self.ytrue = None

    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        # Y = []
        # for x in X:
        #     for layer in self.layers:
        #         x = layer.forward(x)
        #     Y.append(x)
        self.past_input = X
        for layer in self.full_layers:
            X = layer.forward(X)
        return X
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        grad_W_list = []
        grad_b_list = []
        self.ytrue = y_true
        grad_output = self.loss_fn.backward(y_true, y_pred)    
        for layer in reversed(self.full_layers):
            grad_output = layer.backward(grad_output)
            if isinstance(layer, NeuralLayer):
                grad_W_list.append(layer.grad_W)
                grad_b_list.append(layer.grad_b)
        # return [layer.grad_W for layer in self.full_layers if isinstance(layer, NeuralLayer)], [layer.grad_b for layer in self.full_layers if isinstance(layer, NeuralLayer)]
        
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i , (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb
        # print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        # print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b

    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        if isinstance(self.optimizer, NAG):
            original_weights = self.global_weights.copy()
            self.optimizer.compute_lookahead(self.global_weights, self.global_grad)
            Y = self.forward(self.past_input)
            self.backward(self.ytrue, Y)
            self.global_weights[:] = original_weights[:]
            self.optimizer.step(self.global_weights, self.global_grad)
        else:
            self.optimizer.step(self.global_weights, self.global_grad)

    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            loss_epoch = 0.0
            for i in range(0, num_samples, batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                y_pred = self.forward(batch_X)
                loss = self.loss_fn.forward(batch_y, y_pred)
                loss_epoch += loss
                self.backward(batch_y, y_pred)
                self.update_weights()
            loss_epoch /= (num_samples / batch_size)
            
            print(f"Loss for epoch {epoch+1}: {loss_epoch}")

    def evaluate(self, X, y):
        """
        Evaluate the network on given data. return accuracy, and f1 score
        """
        y_pred_logits = self.forward(X)
        loss = self.loss_fn.forward(y, y_pred_logits)
        # Compute accuracy and F1 score (implementation depends on specific requirements)
        y_pred = np.argmax(y_pred_logits, axis=1)
        y_true = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred == y_true)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, average='macro')
        return accuracy, f1
    def _assign_global_params(self):
        idx = 0
        for layer in self.full_layers:
            if isinstance(layer, NeuralLayer):

                w_size = layer.W.size
                b_size = layer.b.size
                # copy weights and bias to global_weights
                self.global_weights[idx:idx+w_size] = layer.W.flatten()
                self.global_weights[idx+w_size:idx+w_size+b_size] = layer.b.flatten()

                # turning global_weights into view for layer weights and bias
                layer.W = self.global_weights[idx:idx+w_size].reshape(layer.W.shape)
                layer.b = self.global_weights[idx+w_size:idx+w_size+b_size].reshape(layer.b.shape)

                # copy gradients to global_grad
                self.global_grad[idx:idx+w_size] = layer.grad_W.flatten()
                self.global_grad[idx+w_size:idx+w_size+b_size] = layer.grad_b.flatten()

                # turning global_grad into view for layer gradients
                layer.grad_W = self.global_grad[idx:idx+w_size].reshape(layer.grad_W.shape)
                layer.grad_b = self.global_grad[idx+w_size:idx+w_size+b_size].reshape(layer.grad_b.shape)
                idx += w_size + b_size
    def sync_global_to_local(self):
        self._assign_global_params()

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W[:] = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b[:] = weight_dict[b_key].copy()