import pandas as pd
import numpy as np


chess = pd.read_csv('./data/games.csv')      #open csv file

#remove not used columns
chess_clean = chess.drop(columns=["id","white_id","black_id","created_at","last_move_at","moves","victory_status","turns","opening_name"])


#add new column with the difference of 2 player's elo
chess_clean["rating_diff"] = chess_clean["white_rating"] - chess_clean["black_rating"]


#reorder important columns
chess_clean = chess_clean[['white_rating', 'black_rating', 'rated','increment_code','opening_eco','opening_ply','rating_diff','winner']]




print(chess_clean.head(10))