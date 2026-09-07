"""Evaluate the saved policy on every dataset state and every possible game."""

from NeuralNetwork import NeuralNetwork
from main import DEFAULT_MODEL, check_winner, get_agent_move, legal_moves
from train import load_dataset


def human_can_force_win(model, human):
    """Exhaustively explore every human response against the deterministic AI."""
    def search(board, player):
        result = check_winner(board)
        if result is not None:
            return result == human
        if player == human:
            return any(search(board[:move] + [player] + board[move + 1:], -player)
                       for move in legal_moves(board))
        move = get_agent_move(model, board, player)
        return search(board[:move] + [player] + board[move + 1:], -player)

    return search([0] * 9, 1)


def main():
    model = NeuralNetwork.load(DEFAULT_MODEL)
    X, targets = load_dataset("tictactoe_moves.csv")
    correct = 0
    for row, target in zip(X, targets):
        board, player = row[:9].astype(int).tolist(), int(row[9])
        prediction = get_agent_move(model, board, player)
        correct += prediction in {int(move) for move in target.split(";")}

    print(f"All-state optimal-move accuracy: {correct / len(X):.2%} ({correct}/{len(X)})")
    print(f"Human can force a win as X: {human_can_force_win(model, 1)}")
    print(f"Human can force a win as O: {human_can_force_win(model, -1)}")


if __name__ == "__main__":
    main()
