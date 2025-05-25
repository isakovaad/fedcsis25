import os
import time
import pandas as pd
import numpy as np


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
    # do not count the king
    count_w = settings.count(piece.upper())
    count_b = settings.count(piece)
    diff_count = count_w - count_b
    total_count = count_w + count_b
    output.extend([count_w, count_b, diff_count, total_count])
    total_pieces_w += count_w
    total_pieces_b += count_b
    total_values_w += count_w * value
    total_values_b += count_b * value
    
  diff_pieces = total_pieces_w - total_pieces_b
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
    
    
  
  