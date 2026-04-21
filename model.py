import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

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


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 12, 15],
    'min_samples_split': [2, 5]
}


#initialize grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42 , class_weight='balanced'), 
    param_grid, 
    cv=3,           
    n_jobs=-1, 
    verbose=2
)

#search on training data 
grid_search.fit(x_train_scaled, y_train)

#get best model found by grid search
best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# mean accuracy of the best model
winner_prediction = best_rf.predict(x_test_scaled)
print(f"Final Accuracy: {accuracy_score(y_test, winner_prediction)}")

print(classification_report(y_test, winner_prediction))

# Stability and Overfitting for Random Forest (Section 1.6)
best_index_rf = grid_search.best_index_
cv_std_rf = grid_search.cv_results_['std_test_score'][best_index_rf]

print(f"Mean CV Accuracy: {grid_search.best_score_:.4f}")
print(f"CV Standard Deviation (Stability): {cv_std_rf:.4f}")
print(f"Train Accuracy: {best_rf.score(x_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {best_rf.score(x_test_scaled, y_test):.4f}")



#LOGISTIC REGRESSION

logistic = LogisticRegression(solver='saga', max_iter=5000, class_weight='balanced')

logistic.fit(x_train_scaled,y_train)

param_grid_lr = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'] 
}

grid_logistic = GridSearchCV(logistic, param_grid_lr, cv=3, n_jobs=-1, verbose=1)
grid_logistic.fit(x_train_scaled, y_train)

#find best model with grid search
best_logistic = grid_logistic.best_estimator_

y_pred_logistic = best_logistic.predict(x_test_scaled)

print("\n--- Logistic Regression Classification Report ---")
print(classification_report(y_test, y_pred_logistic))

# stability, overfitting
best_index_logistic = grid_logistic.best_index_
cv_std_lr = grid_logistic.cv_results_['std_test_score'][best_index_logistic]

print(f"Mean CV Accuracy: {grid_logistic.best_score_:.4f}")
print(f"CV Standard Deviation (Stability): {cv_std_lr:.4f}")
print(f"Train Accuracy: {best_logistic.score(x_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {best_logistic.score(x_test_scaled, y_test):.4f}")


#