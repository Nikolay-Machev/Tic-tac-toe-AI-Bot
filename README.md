# AI Tic-Tac-Toe Bot

I built a neural network from scratch using only NumPy—no PyTorch, TensorFlow, Keras, or autograd—and trained it to play optimal tic-tac-toe. You can play against the trained model directly in your terminal.

**[Dataset available in CSV](tictactoe_moves.csv)**

## How it works

1. **Training data** — I use 4,520 legal, non-terminal tic-tac-toe board states labelled with their game-theoretically optimal moves. When several moves are equally optimal, I retain all of them instead of arbitrarily favoring one.
2. **Model** — I implemented the complete feedforward neural network myself, including He initialization, forward propagation, ReLU activations, softmax, categorical cross-entropy, and backpropagation. The board and current player form the input, and the network treats the possible moves as a nine-class classification problem.
3. **Play** — `main.py` loads the trained weights and lets you play against the model move by move. I mask occupied cells during inference, so the model always chooses a legal move.

The included model selects an optimal move on **99.38% of all 4,520 labelled states**. I also exhaustively traversed every possible human response and confirmed that a human cannot force a win as either X or O against its deterministic policy.

## Project structure

```text
├── NeuralNetwork.py        # Forward pass, backpropagation, prediction, and save/load
├── train.py                # Reproducible dataset loading, training, and evaluation
├── evaluate.py             # Full-state metrics and exhaustive game-tree verification
├── test_project.py         # Dependency-free regression tests
├── tictactoe_moves.csv     # Board states and their minimax-optimal moves
├── models/
│   └── tictactoe.npz       # Saved trained weights
├── main.py                 # Terminal game
├── requirements.txt        # Python dependency list
└── README.md
```

## Setup

```bash
git clone https://github.com/Nikolay-Machev/Tic-tac-toe-AI-Bot.git
cd Tic-tac-toe-AI-Bot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Play against the included model:**

```bash
python main.py
```

Choose X or O, enter the number of an empty square, or enter `q` to quit.

**Train the model yourself:**

```bash
python train.py
```

You can also change the training configuration from the command line:

```bash
python train.py --iterations 6000 --hidden-neurons 192 --learning-rate 0.2 --seed 7
```

## Testing

```bash
python -m unittest -v
python evaluate.py
```

The tests cover win and draw detection, legal moves, occupied-cell masking, probability normalization, and exact model save/load reproduction.

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.
