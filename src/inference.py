"""
Inference Script
Evaluate trained models on test sets
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


def _resolve_path(path_value):
    """Resolve relative paths against this script's directory for CWD-independent behavior."""
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(this, path_value)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default=None)
    parser.add_argument("--model_path", default="best_model.npy")
    parser.add_argument("--config_path", default="best_config.json")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    loaded = np.load(model_path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        loaded = loaded.item()
    if not isinstance(loaded, dict):
        raise ValueError(
            "Expected saved model as a dict of weight tensors (W0, b0, ...)."
        )
    return loaded


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)

    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    y_pred = np.argmax(probs, axis=1).reshape(-1)
    y_true = np.argmax(y_test, axis=1).reshape(-1)

    loss = model.loss_fn.forward(y_test, logits)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



def main():
    """
    Main inference function.
    
    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    args.model_path = _resolve_path(args.model_path)
    args.config_path = _resolve_path(args.config_path)

    # Load config JSON (dictionary with keys: hidden_size, activation, loss, weight_init, etc.)
    with open(args.config_path, "r") as f:
        config = json.load(f)

    dataset_name = args.dataset if args.dataset is not None else config.get("dataset", "mnist")

    # Load dataset
    _, _, X_test, y_test = load_data(dataset_name)

    # Load saved weights (expected to be a dict of W0,b0,...)
    weights = load_model(args.model_path)

    model_cfg = {
    "hidden_size": config.get("hidden_size", []),
    "activation": config.get("activation", "relu"),
    "loss": config.get("loss", "cross_entropy"),
    "weight_init": config.get("weight_init", "xavier"),
    "input_size": X_test.shape[1],
    "output_size": y_test.shape[1],
    "optimizer": None,
}
    model = NeuralNetwork(model_cfg)

    model.set_weights(weights)

    # Evaluate
    results = evaluate_model(model, X_test, y_test)

    print("Evaluation Results:")
    print(f"Dataset: {dataset_name}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}") 
    print("Evaluation complete!")
    return results


if __name__ == '__main__':
    main()
