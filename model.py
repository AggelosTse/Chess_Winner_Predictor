import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

chess = pd.read_csv('./data/games.csv')      #open csv file


#add new column with the difference of 2 player's elo
chess["rating_diff"] = chess["white_rating"] - chess["black_rating"]

#seperate increment code
splitted = chess['increment_code'].str.split('+', expand=True)
chess['base_value'] = splitted[0].astype(int)
chess['increment_value'] = splitted[1].astype(int)


#remove not used columns
chess_clean = chess.drop(columns=["id","white_id","black_id","created_at","last_move_at","moves","victory_status","turns","opening_name",'increment_code'])



#reorder important columns
chess_clean = chess_clean[['white_rating', 'black_rating', 'rated','base_value','increment_value','opening_eco','opening_ply','rating_diff','winner']]

#drops the whole row if it finds a null 
chess_clean = chess_clean.dropna() 

#object to encode data
label_encoder = LabelEncoder()

#encode the non numeric columns
columns = ['rated','opening_eco']

for col in columns:
    chess_clean[col] = label_encoder.fit_transform(chess_clean[col])
    
#encode target variable
chess_clean['winner'] = label_encoder.fit_transform(chess_clean['winner'])


y=chess_clean["winner"]
x=chess_clean.drop("winner", axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

randomfor_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
randomfor_classifier.fit(x_train_scaled, y_train)

winner_prediction = randomfor_classifier.predict(x_test_scaled)

print(f"accuracy: {accuracy_score(y_test,winner_prediction)}")

print(chess_clean.head(10))

