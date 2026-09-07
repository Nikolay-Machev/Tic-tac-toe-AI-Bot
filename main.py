"""Play tic-tac-toe against the trained NumPy model in your terminal."""

import argparse
from pathlib import Path
import numpy as np
from NeuralNetwork import NeuralNetwork

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
         (2, 5, 8), (0, 4, 8), (2, 4, 6))
SYMBOLS = {0: " ", 1: "X", -1: "O"}
DEFAULT_MODEL = Path(__file__).parent / "models" / "tictactoe.npz"


def render_game(board):
    cells = [SYMBOLS[value] if value else str(i + 1) for i, value in enumerate(board)]
    print(f"\n {cells[0]} | {cells[1]} | {cells[2]}\n---+---+---\n {cells[3]} | {cells[4]} | {cells[5]}\n---+---+---\n {cells[6]} | {cells[7]} | {cells[8]}\n")


def check_winner(board):
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0 if all(board) else None


def legal_moves(board):
    return [i for i, value in enumerate(board) if value == 0]


def get_human_move(board, player):
    while True:
        raw = input(f"Your move ({SYMBOLS[player]}) — choose 1-9 or q to quit: ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if raw.isdigit() and 1 <= int(raw) <= 9 and board[int(raw) - 1] == 0:
            return int(raw) - 1
        print("Choose an empty cell from 1 to 9.")


def get_agent_move(model, board, player):
    """Choose the highest-scoring legal move; occupied cells are masked."""
    probabilities = model.predict_proba(np.asarray([board + [player]], dtype=float))[0]
    return max(legal_moves(board), key=lambda move: probabilities[move])


def choose_symbol(value=None):
    choice = (value or input("Choose X or O: ")).strip().upper()
    while choice not in {"X", "O"}:
        choice = input("Please enter X or O: ").strip().upper()
    return 1 if choice == "X" else -1


def play(model, human):
    board, current_player = [0] * 9, 1
    print("\nTic-Tac-Toe — use the numbered cells shown below.")
    render_game(board)
    while check_winner(board) is None:
        if current_player == human:
            move = get_human_move(board, human)
        else:
            move = get_agent_move(model, board, -human)
            print(f"AI ({SYMBOLS[-human]}) chooses {move + 1}.")
        board[move] = current_player
        render_game(board)
        current_player *= -1
    result = check_winner(board)
    print("Draw!" if result == 0 else ("You win!" if result == human else "AI wins!"))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--symbol", choices=("X", "O", "x", "o"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}. Run `python train.py` first.")
    try:
        play(NeuralNetwork.load(args.model), choose_symbol(args.symbol))
    except (KeyboardInterrupt, EOFError):
        print("\nGame ended.")


if __name__ == "__main__":
    main()
