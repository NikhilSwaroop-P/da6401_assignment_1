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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--model_path", default="best_model.npy")
    parser.add_argument("--config_path", default="best_config.json")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    return np.load(model_path, allow_pickle=True).item()


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
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
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

    # Load dataset
    _, _, X_test, y_test = load_data(args.dataset)

    # Load saved weights (expected to be a dict of W0,b0,...)
    weights = load_model(args.model_path)

    # Load config JSON (dictionary with keys: hidden_size, activation, loss, weight_init, etc.)
    hidden_size = []
    with open(args.config_path, "r") as f:
        config = json.load(f)
    for key in weights:
        if key[0] == 'b':
            hidden_size.append((int(key[1:]),weights[key].shape[0]))
    hidden_size.sort()
    hidden_size = [h[1] for h in hidden_size]
    print(hidden_size)
    model_cfg = {
    "hidden_size": hidden_size,
    "activation": config.get("activation", "relu"),
    "loss": config.get("loss", "cross_entropy"),
    "weight_init": config.get("weight_init", "xavier"),
    "input_size": X_test.shape[1],
    "output_size": y_test.shape[1],
    "optimizer": None,
}
    model = NeuralNetwork(model_cfg)
    try:
        model.set_weights(weights)
    except:
        raise ValueError(f"Error in setting weights{model_cfg}, {config}")

    # Evaluate
    results = evaluate_model(model, X_test, y_test)

    print("Evaluation Results:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}") 
    print("Evaluation complete!")
    return results


if __name__ == '__main__':
    main()