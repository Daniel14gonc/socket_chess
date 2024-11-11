import chess

def move_to_index(move, board):
    """
    Convert a chess.Move to a unique index in the range 0-4671.

    Args:
        move (chess.Move): The move to convert.
        board (chess.Board): The current board.

    Returns:
        int: The unique index of the move.

    Raises:
        ValueError: If the move is illegal or the index is out of bounds.
    """
    if type(move) == str:
        move = chess.Move.from_uci(move)
    from_square = move.from_square
    to_square = move.to_square

    NORMAL_MOVE_START = 0
    PROMOTION_MOVE_START = 4096
    CASTLING_MOVE_START = 4352
    EN_PASSANT_MOVE_START = 4356

    if board.is_castling(move):
        castling_map = {
            chess.Move.from_uci("e1g1"): 0,  # White kingside
            chess.Move.from_uci("e1c1"): 1,  # White queenside
            chess.Move.from_uci("e8g8"): 2,  # Black kingside
            chess.Move.from_uci("e8c8"): 3,  # Black queenside
        }
        castling_index = castling_map.get(move, -1)

        if castling_index == -1:
            raise ValueError(f"Invalid castling move: {move}")
        index = CASTLING_MOVE_START + castling_index
        return index

    if board.is_en_passant(move):
        capture_file = chess.square_file(move.to_square)
        if board.turn == chess.WHITE:
            # White en passant captures
            index = EN_PASSANT_MOVE_START + capture_file  # 4356 + file (0-7)
        else:
            # Black en passant captures
            index = EN_PASSANT_MOVE_START + 8 + capture_file  # 4364 + file (0-7)
        if index >= 4372:
            raise ValueError(f"En passant index out of bounds: {index}")
        return index

    if move.promotion:
        promotion_map = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3
        }
        promotion_type = promotion_map.get(move.promotion, None)
        if promotion_type is None:
            raise ValueError(f"Unsupported promotion type: {move.promotion}")
        # Each from_square can have up to 4 promotion types
        promotion_index = PROMOTION_MOVE_START + (from_square * 4) + promotion_type
        if promotion_index >= CASTLING_MOVE_START:
            raise ValueError(f"Promotion index out of bounds: {promotion_index}")
        return promotion_index

    # Normal move
    normal_move_index = NORMAL_MOVE_START + (from_square * 64) + to_square
    if normal_move_index >= PROMOTION_MOVE_START:
        raise ValueError(f"Normal move index out of bounds: {normal_move_index}")
    return normal_move_index

def index_to_move(index, board):
    """
    Convert a unique index back to a chess.Move.

    Args:
        index (int): The index of the move (0-4671).
        board (chess.Board): The current board.

    Returns:
        chess.Move: The corresponding chess.Move.

    Raises:
        ValueError: If the index is out of bounds or does not correspond to a valid move.
    """
    NORMAL_MOVE_START = 0
    PROMOTION_MOVE_START = 4096
    CASTLING_MOVE_START = 4352
    EN_PASSANT_MOVE_START = 4356

    if NORMAL_MOVE_START <= index < PROMOTION_MOVE_START:
        # Normal move
        move_num = index - NORMAL_MOVE_START
        from_square = move_num // 64
        to_square = move_num % 64
        move = chess.Move(from_square, to_square)
        if move not in board.legal_moves:
            raise ValueError(f"Invalid normal move for index {index}: {move}")
        return move

    elif PROMOTION_MOVE_START <= index < CASTLING_MOVE_START:
        # Promotion move
        move_num = index - PROMOTION_MOVE_START
        from_square = move_num // 4
        promotion_type = move_num % 4
        promotion_map = {0: chess.QUEEN, 1: chess.ROOK, 2: chess.BISHOP, 3: chess.KNIGHT}
        promotion_piece = promotion_map.get(promotion_type, chess.QUEEN)

        # Determine the to_square based on the board and promotion
        if board.turn == chess.WHITE:
            to_square = from_square + 8  # Move to the 8th rank
        else:
            to_square = from_square - 8  # Move to the 1st rank

        move = chess.Move(from_square, to_square, promotion=promotion_piece)
        if move not in board.legal_moves:
            raise ValueError(f"Invalid promotion move for index {index}: {move}")
        return move

    elif CASTLING_MOVE_START <= index < EN_PASSANT_MOVE_START:
        # Castling move
        castling_map_reverse = {
            4352: chess.Move.from_uci("e1g1"),  # White kingside
            4353: chess.Move.from_uci("e1c1"),  # White queenside
            4354: chess.Move.from_uci("e8g8"),  # Black kingside
            4355: chess.Move.from_uci("e8c8"),  # Black queenside
        }
        move = castling_map_reverse.get(index, None)
        if move is None or move not in board.legal_moves:
            raise ValueError(f"Invalid castling move for index {index}")
        return move

    elif EN_PASSANT_MOVE_START <= index < 4672:
        # En passant move
        move_num = index - EN_PASSANT_MOVE_START
        from_square = move_num // 8
        to_file = move_num % 8

        # Determine to_square based on from_square and player color
        from_rank = chess.square_rank(from_square)
        if board.turn == chess.WHITE:
            to_rank = from_rank + 1
        else:
            to_rank = from_rank - 1

        to_square = chess.square(to_file, to_rank)
        move = chess.Move(from_square, to_square, chess.Move.EN_PASSANT)
        if move not in board.legal_moves:
            raise ValueError(f"Invalid en passant move for index {index}: {move}")
        return move

    else:
        raise ValueError(f"Move index out of bounds: {index}")
