# Tic-Tac-Toe AI — Neural Network from Scratch

A fully connected neural network built with **NumPy and linear algebra only**—no PyTorch, TensorFlow, Keras, or autograd. It learns from minimax-labelled board positions and then plays against you in the terminal.

## What this project demonstrates

- He initialization, ReLU activations, and a numerically stable softmax
- Categorical cross-entropy and backpropagation implemented by hand
- Multiple equally optimal targets per board position
- Model serialization with NumPy's compressed `.npz` format
- Legal-move masking so the model never selects an occupied cell
- Reproducible training/evaluation and dependency-free unit tests
- Exhaustive game-tree verification against every possible human strategy

The input is the nine-cell board plus the player to move. Cells are encoded as `1` for X, `-1` for O, and `0` for empty. The network produces a probability distribution over all nine moves.

## Quick start

```bash
git clone https://github.com/Nikolay-Machev/Tic-tac-toe-AI-Bot.git
cd Tic-tac-toe-AI-Bot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Choose X or O, then enter the number of an empty square. Use `q` to leave.

## Train it yourself

The checked-in model is ready to play. To reproduce it from the included labelled positions:

```bash
python train.py
```

Training prints held-out **optimal-move accuracy**. A prediction counts as correct when it belongs to the complete set of minimax-optimal moves—not only the dataset's first move.

The included model scores **99.38% across all 4,520 labelled states**. Exhaustive traversal also confirms that a human cannot force a win while playing either X or O against its deterministic policy.

```bash
python train.py --iterations 6000 --hidden-neurons 192 --learning-rate 0.2 --seed 7
python main.py --model models/tictactoe.npz --symbol X
```

## Test

```bash
python -m unittest -v
python evaluate.py
```

Tests cover game results, legal moves, occupied-cell masking, probability normalization, and exact model save/load reproduction.

## Project layout

```text
NeuralNetwork.py      NumPy forward pass, backpropagation, inference, save/load
main.py               Terminal game and legal-move handling
train.py              Dataset loading, training, and evaluation
evaluate.py           Full-state metrics and exhaustive game-tree verification
test_project.py       Standard-library unit tests
tictactoe_moves.csv   4,520 minimax-labelled non-terminal positions
models/               Trained model ready for terminal play
```

## Dataset and honest scope

The CSV contains legal non-terminal board states and every optimal move for the player whose turn it is. Keeping tied optimal moves avoids penalizing one correct choice in favor of another.

Tic-tac-toe is small enough to solve exactly with minimax. This educational model demonstrates how a neural network can approximate that optimal policy from labelled examples; it is not intended to outperform the exact solver.

## License

MIT
