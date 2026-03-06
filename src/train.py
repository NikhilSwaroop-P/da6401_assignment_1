"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
from sklearn.metrics import f1_score
import wandb
import numpy as np
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, RMSProp, Momentum, NAG
import json
def parse_arguments():
    
    parser = argparse.ArgumentParser(description='Train a neural network')
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], required=True)
    parser.add_argument("-e", "--epochs", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True, default=128)
    parser.add_argument("-l", "--loss", choices=["cross_entropy", "MSE"], required=True)
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop"], required=True)
    parser.add_argument("-lr", "--learning_rate", type=float, required=True)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, required=True)
    parser.add_argument("--hidden_size", nargs="+", type=int)
    parser.add_argument("-a", "--activation", choices=["relu", "sigmoid", "tanh"], required=True)
    parser.add_argument("-w_i", "--weight_init", choices=["random", "xavier"], required=True)
    parser.add_argument("-w_p", "--wandb_project", default="DA6401_assignement1")
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    best_f1 = -1
    args = parse_arguments()
    if args.num_layers != len(args.hidden_size):
        raise ValueError("num_layers must match length of hidden_size list")
    
    X_train, y_train, X_test, y_test = load_data(args.dataset)
            
    if args.optimizer == "sgd":
        optimizer = SGD(lr=args.learning_rate,weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = RMSProp(lr=args.learning_rate,weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer = Momentum(lr=args.learning_rate,weight_decay=args.weight_decay)
    elif args.optimizer == "nag":
        optimizer = NAG(lr=args.learning_rate,weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
        
    args.optimizer = optimizer
    args.hidden_size = args.hidden_size
    args.hidden_sizes = args.hidden_size if hasattr(args,"hidden_size") else args.sz
    model = NeuralNetwork(args)
    print("Starting training...")
    model.train(X_train, y_train, args.epochs, args.batch_size)
    y_train_labels = np.argmax(y_train, axis=1)  # Convert one-hot to class labels

    train_acc, train_f1 = model.evaluate(X_train, y_train)
    test_acc, test_f1 = model.evaluate(X_test, y_test)
    if test_f1 > best_f1:
        best_f1 = test_f1
        model_data = {
            "weights": model.get_weights(),
            "config": {
                "hidden_size": args.hidden_size,
                "activation": args.activation,
                "loss": args.loss,
                "weight_init": args.weight_init,
                "optimizer": args.optimizer.__class__.__name__ if args.optimizer is not None else None,
                "learning_rate": args.learning_rate,
                "num_layers": args.num_layers,
                "batch_size": args.batch_size
            }
        }
        np.save("best_model.npy", model_data["weights"])

        with open("best_config.json", "w") as f:
            json.dump(model_data["config"], f)

      
    print("Training complete!")


if __name__ == '__main__':
    main()