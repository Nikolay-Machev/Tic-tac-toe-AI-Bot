"""Dependency-free regression tests (run with `python -m unittest`)."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from NeuralNetwork import NeuralNetwork
from main import check_winner, get_agent_move, legal_moves


class GameTests(unittest.TestCase):
    def test_results(self):
        self.assertEqual(check_winner([1, 1, 1, 0, -1, 0, -1, 0, 0]), 1)
        self.assertEqual(check_winner([1, -1, 1, 1, -1, -1, -1, 1, 1]), 0)
        self.assertIsNone(check_winner([1, 0, 0, 0, -1, 0, 0, 0, 0]))

    def test_legal_moves(self):
        self.assertEqual(legal_moves([1, 0, -1, 0, 0, 0, 0, 0, 0]),
                         [1, 3, 4, 5, 6, 7, 8])

    def test_agent_masks_occupied_cells(self):
        model = NeuralNetwork(n_iters=1, random_state=1)
        model.fit(np.zeros((2, 10)), [0, 1])
        board = [1, -1, 1, -1, 0, 1, -1, 1, -1]
        self.assertEqual(get_agent_move(model, board, 1), 4)


class NetworkTests(unittest.TestCase):
    def test_prediction_and_save_load_round_trip(self):
        X = np.eye(3)
        model = NeuralNetwork(layer_neurons=8, final_neurons=3,
                              n_iters=300, lr=0.2, random_state=7).fit(X, [0, 1, 2])
        before = model.predict_proba(X)
        self.assertEqual(before.shape, (3, 3))
        self.assertTrue(np.allclose(before.sum(axis=1), 1.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            after = NeuralNetwork.load(path).predict_proba(X)
        self.assertTrue(np.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
