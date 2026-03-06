"""
Inference Script
Evaluate trained models on test sets
"""
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,confusion_matrix
import wandb
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--dataset",
    choices=["mnist","fashion_mnist"],
    required=True)

    parser.add_argument("--model_path", required=True)

    parser.add_argument("--config_path", required=True)

    parser.add_argument("-b","--batch_size", type=int, default=128)
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

    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    loss = model.loss.forward(logits, y_test)

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
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Build a small cli-like object expected by NeuralNetwork
    class DummyArgs:
        pass

    cli = DummyArgs()
    # map config keys to the cli object used by NeuralNetwork
    # Accept both 'hidden_size' or maybe 'hidden_sizes' inside config
    cli.hidden_size = config.get("hidden_size", config.get("hidden_sizes", config.get("sz", [])))
    cli.activation = config.get("activation", "relu")
    cli.loss = config.get("loss", "cross_entropy")
    cli.weight_init = config.get("weight_init", "random")
    cli.num_layers = config.get("num_layers", len(cli.hidden_size) if cli.hidden_size else 0)
    cli.batch_size = config.get("batch_size", args.batch_size)
    # optimizer not needed for inference, but NeuralNetwork expects an attribute
    cli.optimizer = None

    # Initialize model and set weights
    model = NeuralNetwork(cli)
    model.set_weights(weights)

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