import pandas as pd
import numpy as np


chess = pd.read_csv('./data/games.csv')      #open csv file

chess_clean = chess.drop(columns=["id","white_id","black_id","created_at","last_move_at","moves",])
print(chess_clean.head(10))