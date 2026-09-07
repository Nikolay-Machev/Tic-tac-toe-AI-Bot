"""A small fully connected classifier implemented only with NumPy."""

from pathlib import Path
from typing import Iterable, Sequence
import numpy as np


class NeuralNetwork:
    """Multi-layer perceptron with ReLU activations and softmax output."""

    def __init__(self, hidden_layers=1, lr=0.1, layer_neurons=128,
                 n_iters=2_000, final_neurons=9, random_state=42):
        if hidden_layers < 1 or layer_neurons < 1 or final_neurons < 2:
            raise ValueError("invalid layer size")
        self.hidden_layers, self.lr = hidden_layers, lr
        self.neurons = self.nuerons = layer_neurons  # old-name compatibility
        self.n_iters, self.final_neurons = n_iters, final_neurons
        self.random_state = random_state
        self.weights, self.biases, self.loss_history = [], [], []

    @property
    def bias(self):
        return self.biases

    @bias.setter
    def bias(self, value):
        self.biases = value

    def _init_weights(self, input_features):
        rng = np.random.default_rng(self.random_state)
        sizes = [input_features] + [self.neurons] * self.hidden_layers + [self.final_neurons]
        self.weights, self.biases = [], []
        for index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt((1.0 if index == len(sizes) - 2 else 2.0) / fan_in)
            self.weights.append(rng.standard_normal((fan_out, fan_in)) * scale)
            self.biases.append(np.zeros((fan_out, 1)))

    def fit(self, X: Sequence[Sequence[float]], y: Iterable, multi=False):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not len(X):
            raise ValueError("X must be a non-empty 2D array")
        rows = list(y)
        targets = self._multi_hot(rows) if multi else self._one_hot(np.asarray(rows, dtype=int))
        if targets.shape[1] != len(X):
            raise ValueError("X and y must have the same length")
        self._init_weights(X.shape[1])
        self.loss_history = []
        for iteration in range(self.n_iters):
            activations, pre_activations = self._forward(X)
            probabilities = self.softmax(activations[-1])
            if iteration == 0 or (iteration + 1) % 100 == 0 or iteration == self.n_iters - 1:
                clipped = np.clip(probabilities, 1e-12, 1.0)
                self.loss_history.append(float(-np.sum(targets * np.log(clipped)) / len(X)))
            self._backward(activations, pre_activations, probabilities, targets, len(X))
        return self

    def _forward(self, X):
        if not self.weights:
            raise RuntimeError("fit or load the model before prediction")
        activations, pre_activations = [X.T], []
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            z = weights @ activations[-1] + bias
            pre_activations.append(z)
            activations.append(z if index == len(self.weights) - 1 else self.relu(z))
        return activations, pre_activations

    def _backward(self, activations, pre_activations, probabilities, targets, sample_count):
        delta, gradients = probabilities - targets, []
        for index in reversed(range(len(self.weights))):
            gradients.append((delta @ activations[index].T / sample_count,
                              np.mean(delta, axis=1, keepdims=True)))
            if index:
                delta = (self.weights[index].T @ delta) * (pre_activations[index - 1] > 0)
        for index, (d_weights, d_bias) in enumerate(reversed(gradients)):
            self.weights[index] -= self.lr * d_weights
            self.biases[index] -= self.lr * d_bias

    @staticmethod
    def relu(values):
        return np.maximum(0.0, values)

    ReLU = relu

    def _one_hot(self, y):
        if np.any((y < 0) | (y >= self.final_neurons)):
            raise ValueError("target class is outside output range")
        result = np.zeros((self.final_neurons, len(y)))
        result[y, np.arange(len(y))] = 1.0
        return result

    def _multi_hot(self, labels):
        result = np.zeros((self.final_neurons, len(labels)))
        for column, label in enumerate(labels):
            moves = ([int(move) for move in label.split(";")] if isinstance(label, str)
                     else [int(label)] if np.isscalar(label) else [int(move) for move in label])
            if not moves or any(move < 0 or move >= self.final_neurons for move in moves):
                raise ValueError(f"invalid target moves: {moves}")
            result[moves, column] = 1.0 / len(moves)
        return result

    @staticmethod
    def softmax(logits):
        shifted = logits - np.max(logits, axis=0, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=0, keepdims=True)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        activations, _ = self._forward(X)
        return self.softmax(activations[-1]).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path):
        if not self.weights:
            raise RuntimeError("cannot save an unfitted model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, hidden_layers=self.hidden_layers,
            neurons=self.neurons, final_neurons=self.final_neurons,
            random_state=-1 if self.random_state is None else self.random_state,
            **{f"W{i}": value for i, value in enumerate(self.weights)},
            **{f"b{i}": value for i, value in enumerate(self.biases)})

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            seed = int(data["random_state"]) if "random_state" in data else 42
            model = cls(hidden_layers=int(data["hidden_layers"]),
                layer_neurons=int(data["neurons"]), final_neurons=int(data["final_neurons"]),
                random_state=None if seed == -1 else seed)
            model.weights = [data[f"W{i}"].copy() for i in range(model.hidden_layers + 1)]
            model.biases = [data[f"b{i}"].copy() for i in range(model.hidden_layers + 1)]
        return model
