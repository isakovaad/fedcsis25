'''
Number of training samples: 3888765, testing samples: 22823
'''

import os
import time
import numpy as np
import pandas as pd

from fenparser import FenParser

# list of pieces in a board
pieces_black = ['q', 'r', 'b', 'n', 'p']
pieces_white = ['Q', 'R', 'B', 'N', 'P']
pieces_value = [9, 7, 3, 3, 1]
pieces_complexity = [12, 7, 5, 4, 1]

'''
generate features from the initial 'FEN' value
'''
def generate_fen_features(x):
    output = []
    items = x.split()

    total_pieces_w, total_pieces_b = 0, 0
    total_values_w, total_values_b = 0, 0

    settings = items[0]
    for piece, value in zip(pieces_black, pieces_value):
        # do not count the King
        count_w = settings.count(piece.upper())
        count_b = settings.count(piece)
        diff_count = count_w - count_b
        total_count = count_w + count_b
        output.extend([count_w, count_b, diff_count, total_count])
        total_pieces_w += count_w
        total_pieces_b += count_b
        total_values_w += count_w * value
        total_values_b += count_b * value

    qdiff_pieces = total_pieces_w - total_pieces_b
    total_pieces = total_pieces_b + total_pieces_w
    diff_values = total_values_w - total_values_b
    total_values = total_values_w + total_values_b
    output.extend([total_pieces_w, total_pieces_b, diff_pieces, total_pieces])
    output.extend([total_values_w, total_values_b, diff_values, total_values])

    # who makes the next move
    active = 1 if items[1] == "w" else 0

    # castling ability
    castling_wk = 1 if "K" in items[2] else 0
    castling_wq = 1 if "Q" in items[2] else 0
    castling_bk = 1 if "k" in items[2] else 0
    castling_bq = 1 if "q" in items[2] else 0
    total_castling_w = castling_wk + castling_wq
    total_castling_b = castling_bk + castling_bq
    diff_castling = total_castling_w - total_castling_b
    total_castling = total_castling_w + total_castling_b

    # en passant
    en_passant = 1 if items[3] != "-" else 0

    # half and full move (half move is used in 50 move rule to determine tie of a game)
    halfmove_clock = int(items[4])
    fullmove_num = int(items[5])

    output.extend([
        active, en_passant,
        castling_wk, castling_wq, total_castling_w,
        castling_bk, castling_bq, total_castling_b,
        diff_castling, total_castling,
        halfmove_clock, fullmove_num
    ])
    return pd.Series(output)

'''
convert fen to dict_board
'''
def convert_fen_dict(fen):
    board_fen = FenParser(fen).parse()

    board_dict = {
        'a': {'1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''},
        'b': {'1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''},
        'c': {'1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''},
        'd': {'1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''},
        'e': {'1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''},
        'f': {'1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''},
        'g': {'1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''},
        'h': {'1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''}
    }

    for i, row, in zip(range(8), ['8', '7', '6', '5', '4', '3', '2', '1']):
        for j, col in zip(range(8), ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']):
            if board_fen[i][j] != ' ':
                board_dict[col][row] = board_fen[i][j]
            else:
                board_dict[col][row] = ''

    return board_dict

'''
update board and features given a move
'''
def update_board(move, board, features_dict, active):
    possible_moves = count_possible_moves_all(active, board)
    features_dict["possible_moves"] += possible_moves

    selected_piece = board[move[0]][move[1]].lower()
    if selected_piece == "k":
        features_dict["complexity"] += possible_moves
    else:
        features_dict["complexity"] += possible_moves * pieces_complexity[pieces_black.index(selected_piece)]

    if board[move[0]][move[1]] == '':
        print(f"ERROR: {move}")
        print(board)
        exit(-1)

    # update move counting of the piece
    features_dict[f"move_{board[move[0]][move[1]]}"] += 1

    # update kill counting of the piece
    if board[move[2]][move[3]] != '':
        features_dict[f"kill_{board[move[2]][move[3]]}"] += 1

    # check and update white castling
    if board[move[0]][move[1]] == 'K' and move == 'e1g1':
        if board['h']['1'] != 'R':
            print(f"ERROR: {move}")
            print(board)
            exit(-1)
        board['f']['1'] = 'R'
        board['h']['1'] = ''
        features_dict["castled_wk"] += 1

    if board[move[0]][move[1]] == 'K' and move == 'e1c1':
        if board['a']['1'] != 'R':
            print(f"ERROR: {move}")
            print(board)
            exit(-1)
        board['d']['1'] = 'R'
        board['a']['1'] = ''
        features_dict["castled_wq"] += 1

    # check and update black castling
    if board[move[0]][move[1]] == 'k' and move == 'e8g8':
        if board['h']['8'] != 'r':
            print(f"ERROR: {move}")
            print(board)
            exit(-1)
        board['f']['8'] = 'r'
        board['h']['8'] = ''
        features_dict["castled_bk"] += 1

    if board[move[0]][move[1]] == 'k' and move == 'e8c8':
        if board['a']['8'] != 'r':
            print(f"ERROR: {move}")
            print(board)
            exit(-1)
        board['d']['8'] = 'r'
        board['a']['8'] = ''
        features_dict["castled_bq"] += 1

    # update board after the move
    board[move[2]][move[3]] = board[move[0]][move[1]]
    board[move[0]][move[1]] = ''

    return board, features_dict

'''
count the number of possible moves of a piece
'''
def count_possible_moves_piece(piece):
    if piece.lower() == "p":
        return 2
    elif piece.lower() == "n":
        return 8
    elif piece.lower() == "b":
        return 11
    elif piece.lower() == "r":
        return 14
    elif piece.lower() == "q":
        return 20
    else: #piece.lower() == "k:
        return 6

'''
count all possible moves at one time
'''
def count_possible_moves_all(active, board):
    all_possible_moves = 0
    if active == 'w':
        for col in board.keys():
            for row in board[col].keys():
                if board[col][row] in pieces_white:
                    all_possible_moves += count_possible_moves_piece(board[col][row])
    else:
        for col in board.keys():
            for row in board[col].keys():
                if board[col][row] in pieces_black:
                    all_possible_moves += count_possible_moves_piece(board[col][row])
    return all_possible_moves

'''
generate features from the 'Moves' value
'''
def generate_move_features(x, fen):
    active = fen.split()[1]
    board = convert_fen_dict(fen)
    moves = x.split()

    # create a dict of features
    features_dict = dict()
    for piece in pieces_white + pieces_black:
        features_dict[f"move_{piece}"] = 0
        features_dict[f"kill_{piece}"] = 0

    # add move of kings
    features_dict["move_K"] = 0
    features_dict["move_k"] = 0

    # add castling
    features_dict["castled_wk"] = 0
    features_dict["castled_wq"] = 0
    features_dict["castled_bk"] = 0
    features_dict["castled_bq"] = 0

    # add move possibility
    features_dict["possible_moves"] = 0
    features_dict["complexity"] = 0

    num_of_moves = len(moves)
    for move in moves:
        board, features_dict = update_board(move, board, features_dict, active)
        active = "w" if active == "b" else "w"

    # add total moves for each piece both sides and total kill each side
    for piece in pieces_black:
        features_dict[f"move_total_{piece}"] = features_dict[f"move_{piece}"] + features_dict[f"move_{piece.upper()}"]

    # post processing to add extra features
    features_dict["kill_total_w"] = np.sum([features_dict[f"kill_{piece}"] for piece in pieces_white])
    features_dict["kill_total_b"] = np.sum([features_dict[f"kill_{piece}"] for piece in pieces_black])
    features_dict["kill_value_w"] = np.sum([features_dict[f"kill_{piece}"] * value for piece, value in zip(pieces_white, pieces_value)])
    features_dict["kill_value_b"] = np.sum([features_dict[f"kill_{piece}"] * value for piece, value in zip(pieces_black, pieces_value)])

    output = [num_of_moves]
    for piece in pieces_white + pieces_black:
        output.extend([features_dict[f"move_{piece}"], features_dict[f"kill_{piece}"]])
    for piece in pieces_black:
        output.append(features_dict[f"move_total_{piece}"])
    output.extend([features_dict["move_K"], features_dict["move_k"]])
    output.extend([features_dict["castled_wk"], features_dict["castled_wq"]])
    output.extend([features_dict["castled_bk"], features_dict["castled_bq"]])
    output.extend([features_dict["kill_total_w"], features_dict["kill_total_b"]])
    output.extend([features_dict["kill_value_w"], features_dict["kill_value_b"]])
    output.extend([features_dict["possible_moves"], features_dict["complexity"]])
    return pd.Series(output)

def generate_features(limit_training_samples=10000000):
    fen_features = []
    for piece in pieces_black:
        fen_features.extend([f"count_w{piece}", f"count_b{piece}", f"diff_{piece}", f"total_{piece}"])
    fen_features.extend(["total_pieces_w", "total_pieces_b", "diff_pieces", "total_pieces"])
    fen_features.extend(["total_values_w", "total_values_b", "diff_values", "total_values"])
    fen_features.extend([
        "active", "en_passant",
        "castling_wk", "castling_wq", "total_castling_w",
        "castling_bk", "castling_bq", "total_castling_b",
        "diff_castling", "total_castling",
        "halfmove_clock", "fullmove_num"
    ])
    moves_features = [
        "num_of_moves"
    ]
    for piece in pieces_black:
        moves_features.extend([f"move_w{piece}", f"kill_w{piece}"])
    for piece in pieces_black:
        moves_features.extend([f"move_b{piece}", f"kill_b{piece}"])
    moves_features.extend(["move_wk", "move_bk"])
    moves_features.extend(["castled_wk", "castled_wq"])
    moves_features.extend(["castled_bk", "castled_bq"])
    for piece in pieces_black:
        moves_features.append(f"move_total_{piece}")
    moves_features.extend(["kill_total_w", "kill_total_b"])
    moves_features.extend(["kill_value_w", "kill_value_b"])
    moves_features.extend(["possible_moves", "complexity"])

    train_saving_columns = ["PuzzleId"] + fen_features + moves_features + ["Rating"]
    test_saving_columns = ["PuzzleId"] + fen_features + moves_features

    # generate train features
    mode = "w"
    header = True
    train_samples = 0
    start_time = time.time()
    for df_chunk in pd.read_csv(os.path.join("input", "lichess_db_puzzle.csv"), chunksize=100000):
        df_chunk[fen_features] = df_chunk.apply(lambda x: generate_fen_features(x["FEN"]), axis=1)
        df_chunk[moves_features] = df_chunk.apply(lambda x: generate_move_features(x["Moves"], x["FEN"]), axis=1)
        df_chunk[train_saving_columns].to_csv(os.path.join("input", "train_features.csv"), header=header, mode=mode, index=False)
        train_samples += len(df_chunk)
        print(f"Finished processing {train_samples} samples in {round(time.time() - start_time, 2)} seconds")
        if header:
            header = False
            mode = "a"

        if train_samples >= limit_training_samples:
            break

    # generate test features
    df_test = pd.read_csv(os.path.join("input", "test_data_set.csv"))
    df_test[fen_features] = df_test.apply(lambda x: generate_fen_features(x["FEN"]), axis=1)
    df_test[moves_features] = df_test.apply(lambda x: generate_move_features(x["Moves"], x["FEN"]), axis=1)
    df_test[test_saving_columns].to_csv(os.path.join("input", "test_features.csv"), index=False)
    print(f"Total number of records in df_train {train_samples}, df_test {len(df_test)}")

if __name__ == "__main__":
    fen = 'r3k1nr/pbp2p2/1pNbpq2/7p/3P2p1/3BPP2/PPQB2PP/R3K2R w KQkq - 3 14'
    moves = ' '.join(['e1g1', 'd6h2', 'g1h2', 'f6h4', 'h2g1', 'g4g3', 'f1e1', 'h4h2', 'g1f1', 'h2h1', 'f1e2', 'h1g2', 'e2d1', 'g2f3', 'd3e2', 'f3c6', 'c2c6', 'b7c6'])
    #convert_fen_dict(fen)
    generate_move_features(moves, fen)
    #generate_features()