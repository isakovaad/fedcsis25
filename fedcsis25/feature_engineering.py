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
    
    return pd.Series(output)
    
    
  
  