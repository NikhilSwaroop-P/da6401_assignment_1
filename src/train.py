"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import os
import sys

this = os.path.dirname(os.path.abspath(__file__))
thisF = os.path.dirname(this)
project_root = os.path.dirname(thisF)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import numpy as np
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, RMSProp, Momentum, NAG


def _resolve_path(path_value):
    """Resolve relative paths against this script's directory for CWD-independent behavior."""
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(this, path_value)


def _read_existing_best_f1(config_path):
    """Read prior saved test F1 if present; return -inf when unavailable."""
    if not os.path.exists(config_path):
        return float("-inf")
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return float(cfg.get("test_f1", float("-inf")))
    except Exception:
        return float("-inf")

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
    parser.add_argument("-l", "--loss", choices=["cross_entropy", "mse"], required=True)
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop"], required=True)
    parser.add_argument("-lr", "--learning_rate", type=float, required=True)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, required=True)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, required=True)
    parser.add_argument("-a", "--activation", choices=["relu", "sigmoid", "tanh"], required=True)
    parser.add_argument("-w_i", "--weight_init", choices=["random", "xavier"], required=True)
    parser.add_argument("-w_p", "--wandb_project", default="DA6401_assignement1")
    parser.add_argument("--model_save_path", default="src/best_model.npy")
    parser.add_argument("--config_save_path", default="src/best_config.json")
    parser.add_argument(
        "--overwrite_if_worse",
        action="store_true",
        help="Overwrite existing artifacts even when current test_f1 is lower.",
    )
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    best_f1 = -1
    args = parse_arguments()
    args.model_save_path = _resolve_path(args.model_save_path)
    args.config_save_path = _resolve_path(args.config_save_path)
    optimizer_name = args.optimizer
    if args.num_layers != len(args.hidden_size):
        raise ValueError("num_layers must match length of hidden_size list")
    
    X_train, y_train, X_test, y_test = load_data(args.dataset)
            
    if args.optimizer == "sgd":
        optimizer = SGD(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = RMSProp(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer = Momentum(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "nag":
        optimizer = NAG(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
        
    args.optimizer = optimizer
    args.hidden_size = args.hidden_size
    # args.hidden_sizes = args.hidden_size if hasattr(args,"hidden_size") else args.sz
    model = NeuralNetwork(args)
    print("Starting training...")
    model.train(X_train, y_train, args.epochs, args.batch_size)

    # train_acc, train_f1 = model.evaluate(X_train, y_train)
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
                "optimizer": optimizer_name,
                "learning_rate": args.learning_rate,
                "num_layers": args.num_layers,
                "batch_size": args.batch_size,
                "dataset": args.dataset,
                "weight_decay": args.weight_decay,
                "input_size": model.input_size,
                "output_size": model.output_size,
                # 'train_accuracy': train_acc,
                # 'train_f1': train_f1,
                'test_accuracy': test_acc,
                'test_f1': test_f1
            }
        }
    previous_best_f1 = _read_existing_best_f1(args.config_save_path)
    should_save = args.overwrite_if_worse or (test_f1 >= previous_best_f1)

    model_dir = os.path.dirname(args.model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    config_dir = os.path.dirname(args.config_save_path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)

    if should_save:
        np.save(args.model_save_path, model_data["weights"])
        with open(args.config_save_path, "w") as f:
            json.dump(model_data["config"], f)
        print(
            f"Saved model/config to {args.model_save_path} and {args.config_save_path} "
            f"(test_f1={test_f1:.4f}, previous_best_f1={previous_best_f1:.4f})"
        )
    else:
        print(
            f"Skipped saving because current test_f1={test_f1:.4f} < previous_best_f1={previous_best_f1:.4f}. "
            "Use --overwrite_if_worse to force save."
        )

      
    print("Training complete!")


if __name__ == '__main__':
    main()
