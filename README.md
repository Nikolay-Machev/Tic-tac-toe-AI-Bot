# Tic-Tac-Toe AI — Neural Network from Scratch

I built this fully connected neural network using **NumPy and linear algebra only**—no PyTorch, TensorFlow, Keras, or autograd. I trained it on minimax-labelled board positions, and you can play against the resulting model directly in the terminal.

## What I implemented

- He initialization, ReLU activations, and a numerically stable softmax
- Categorical cross-entropy and backpropagation implemented by hand
- Multiple equally optimal targets per board position
- Model serialization with NumPy's compressed `.npz` format
- Legal-move masking so the model never selects an occupied cell
- Reproducible training/evaluation and dependency-free unit tests
- Exhaustive game-tree verification against every possible human strategy

I represent each input using the nine-cell board plus the player to move. I encode X as `1`, O as `-1`, and an empty cell as `0`. The network produces a probability distribution over all nine possible moves.

## Quick start

```bash
git clone https://github.com/Nikolay-Machev/Tic-tac-toe-AI-Bot.git
cd Tic-tac-toe-AI-Bot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Choose X or O, then enter the number of an empty square. I also added `q` as a quick way to leave the game.

## Train it yourself

I included a trained model so the project is ready to play immediately. To reproduce it from the labelled positions:

```bash
python train.py
```

My training script reports held-out **optimal-move accuracy**. I count a prediction as correct when it belongs to the complete set of minimax-optimal moves, rather than requiring it to match only the dataset's first move.

The model I included scores **99.38% across all 4,520 labelled states**. I also exhaustively traversed the game tree and confirmed that a human cannot force a win as either X or O against its deterministic policy.

```bash
python train.py --iterations 6000 --hidden-neurons 192 --learning-rate 0.2 --seed 7
python main.py --model models/tictactoe.npz --symbol X
```

## Test

```bash
python -m unittest -v
python evaluate.py
```

The tests I added cover game results, legal moves, occupied-cell masking, probability normalization, and exact model save/load reproduction.

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

I train against a CSV containing legal non-terminal board states and every optimal move available to the player whose turn it is. By keeping tied optimal moves, I avoid penalizing the network for choosing one correct move instead of another.

Tic-tac-toe is small enough to solve exactly with minimax. I built this as an educational model to demonstrate how a neural network can approximate an optimal policy from labelled examples, not to claim that it can outperform the exact solver.

## License

MIT
