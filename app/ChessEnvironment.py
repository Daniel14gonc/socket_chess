import chess


class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        """Reset the board to the starting position."""
        self.board.reset()
        return self._get_state()

    def step(self, action):
        """
        Apply the given move and return the new state, reward, and whether the game is done.

        Args:
        action (chess.Move): The move to apply

        Returns:
        tuple: (state, reward, done, info)
        """
        if type(action) == str:
           action = chess.Move.from_uci(action)
        if action not in self.board.legal_moves:
            raise ValueError("Illegal move")

        self.board.push(action)

        done = self.board.is_game_over()
        reward = self._get_reward()
        state = self._get_state()
        info = self._get_game_info()

        return state, reward, done, info

    def get_legal_moves(self):
        """Return a list of legal moves in the current position."""
        return list(self.board.legal_moves)

    def _get_state(self):
        """
        Convert the current board state to a format suitable for the neural network.

        Returns:
        numpy.array: An 8x8x12 binary tensor representing the board state
        """
        import numpy as np

        state = np.zeros((8, 8, 12), dtype=np.float32)

        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                color = int(piece.color)
                piece_type = piece_types.index(piece.piece_type)
                state[i // 8, i % 8, piece_type + 6*color] = 1

        return state

    def _get_reward(self):
        """
        Calculate the reward for the current board state.

        Returns:
        float: 1 for a win, -1 for a loss, 0 for draw or ongoing game
        """
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        elif self.board.is_game_over():
            return 0  # Draw
        else:
            return 0  # Ongoing game

    def _get_game_info(self):
        """
        Get detailed information about the current game state.

        Returns:
        dict: Information about the game state
        """
        info = {
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'is_insufficient_material': self.board.is_insufficient_material(),
            'is_fifty_moves': self.board.is_fifty_moves(),
            'is_repetition': self.board.is_repetition(),
            'fullmove_number': self.board.fullmove_number,
            'halfmove_clock': self.board.halfmove_clock,
            'turn': 'White' if self.board.turn == chess.WHITE else 'Black'
        }
        return info

    def copy(self):
        """Create a deep copy of the environment."""
        new_env = ChessEnvironment()
        new_env.board = self.board.copy()
        return new_env

    def render(self):
        """Display the current state of the board."""
        print(self.board)

    def is_game_over(self):
        """Check if the game has ended."""
        return self.board.is_game_over()

    def get_board(self):
        return self.board

    def get_board_from_state(self, state):
        """
        Recreate the chess board from the state tensor.

        Args:
        state (numpy.array): An 8x8x12 binary tensor representing the board state

        Returns:
        chess.Board: A board object representing the current state
        """
        new_board = chess.Board.empty()  # Create an empty board

        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

        for i in range(64):
            row, col = divmod(i, 8)
            for j in range(12):
                if state[row, col, j] == 1:
                    piece_type = piece_types[j % 6]
                    color = chess.WHITE if j < 6 else chess.BLACK
                    square = chess.square(col, 7 - row)  # Board indexing is flipped
                    new_board.set_piece_at(square, chess.Piece(piece_type, color))
                    break

        return new_board
