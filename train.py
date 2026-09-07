"""Train and evaluate the model without an ML framework."""

import argparse
import csv
from pathlib import Path
import numpy as np
from NeuralNetwork import NeuralNetwork

ROOT = Path(__file__).parent


def load_dataset(path):
    features, targets = [], []
    with Path(path).open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            features.append([float(row[f"cell{i}"]) for i in range(9)] +
                            [float(row["player_to_move"])])
            targets.append(row["best_moves_all"])
    return np.asarray(features), targets


def split_dataset(X, y, test_size=0.2, seed=42):
    indices = np.random.default_rng(seed).permutation(len(X))
    cut = int(len(X) * (1.0 - test_size))
    train, test = indices[:cut], indices[cut:]
    return X[train], X[test], [y[i] for i in train], [y[i] for i in test]


def optimal_move_accuracy(predictions, targets):
    correct = [int(prediction) in {int(move) for move in target.split(";")}
               for prediction, target in zip(predictions, targets)]
    return float(np.mean(correct))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "tictactoe_moves.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "tictactoe.npz")
    parser.add_argument("--iterations", type=int, default=4_000)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--hidden-neurons", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = split_dataset(X, y, seed=args.seed)
    model = NeuralNetwork(layer_neurons=args.hidden_neurons, lr=args.learning_rate,
                          n_iters=args.iterations, random_state=args.seed)
    model.fit(X_train, y_train, multi=True)
    score = optimal_move_accuracy(model.predict(X_test), y_test)
    model.save(args.output)
    print(f"Samples: {len(X):,} ({len(X_train):,} train / {len(X_test):,} test)")
    print(f"Final cross-entropy: {model.loss_history[-1]:.4f}")
    print(f"Optimal-move test accuracy: {score:.2%}")
    print(f"Saved model: {args.output}")


if __name__ == "__main__":
    main()
