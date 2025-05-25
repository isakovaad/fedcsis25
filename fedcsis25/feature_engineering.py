import os
import time
import pandas as pd
import numpy as np

feature_names = []

# 5 piece‐types × 4 stats each = 20
for p in ['Q','R','B','N','P']:
    feature_names += [
      f"{p}_w_count",    # how many White p’s
      f"{p}_b_count",    # how many Black p’s
      f"{p}_diff",       # white minus black
      f"{p}_total"       # white + black
    ]

# aggregates: piece counts (4), material values (4) → 8 more
feature_names += [
    "total_pieces_w",   # sum of all White non‐king pieces
    "total_pieces_b",   # sum of all Black non‐king pieces
    "diff_pieces",      # white minus black piece‐count
    "total_pieces",     # total non‐king pieces on board
    "total_val_w",      # material value of White’s pieces
    "total_val_b",      # material value of Black’s pieces
    "diff_values",      # white minus black material
    "total_values"      # total material on board
]

# move‐flags, castling, en passant, clocks → 12 more
feature_names += [
    "active",           # 1 if White to move, else 0
    "en_passant",       # 1 if an en‐passant square exists
    "castling_wk",      # White can castle kingside?
    "castling_wq",      # White can castle queenside?
    "total_castling_w", # White’s total castling rights (0–2)
    "castling_bk",      # Black kingside?
    "castling_bq",      # Black queenside?
    "total_castling_b", # Black’s total (0–2)
    "diff_castling",    # white minus black rights
    "total_castling",   # total castling rights on both sides
    "halfmove_clock",   # half‐move clock for 50‐move rule
    "fullmove_number"   # move number in the game
]


# list of pieces in a board
pieces_black = ['q', 'r', 'b', 'n', 'p']
pieces_white = ['Q', 'R', 'B', 'N', 'P']
pieces_value = [9, 7, 3, 3, 1]
pieces_complexity = [12, 7, 5, 4, 1]

'''
generate features from the initial 'FEN' value
'''
def hello_world():
  return "hello!"


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
    
  return pd.Series(output, index=feature_names)
    
    
  
  