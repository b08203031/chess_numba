# main.py

from chess_engine.board import Board

def main():
    """
    Main function to test the chess engine components.
    """
    # Create a new board instance
    board = Board()

    # Print the initial board state
    board.print_board()

if __name__ == "__main__":
    main()
