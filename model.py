import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

chess = pd.read_csv('./data/games.csv')      #open csv file

#drop draws
chess = chess[chess['winner'] != 'draw'].copy()

#add new column with the difference of 2 player's elo
chess["rating_diff"] = chess["white_rating"] - chess["black_rating"]

#seperate increment code
splitted = chess['increment_code'].str.split('+', expand=True)
chess['base_value'] = splitted[0].astype(int)
chess['increment_value'] = splitted[1].astype(int)

#reorder important columns
chess_clean = chess[['white_rating', 'black_rating', 'rated','base_value','increment_value','opening_eco','opening_ply','rating_diff','winner']]

#drops the whole row if it finds a null 
chess_clean = chess_clean.dropna() 

#encode the non numeric columns
label_encoder = LabelEncoder()
columns = ['rated','opening_eco']

for col in columns:
    chess_clean[col] = label_encoder.fit_transform(chess_clean[col])
    
print("Raw Class Counts")
print(chess_clean['winner'].value_counts())
print("Class Percentages")
print(chess_clean['winner'].value_counts(normalize=True) * 100)

#encode target variable for binary classification
chess_clean['winner'] = chess_clean['winner'].map({'white': 0, 'black': 1})

#train test split
y=chess_clean["winner"]
x=chess_clean.drop("winner", axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#scaling variables
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#RANDOM FOREST CLASSIFIER 

param_grid_randomForest = {
    'n_estimators': [100, 200],
    'max_depth': [10, 12, 15],
    'min_samples_split': [2, 5]
}

#initialize grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    param_grid_randomForest, 
    cv=3,           
    n_jobs=-1, 
    verbose=1
)

#search on training data 
grid_search.fit(x_train_scaled, y_train)

#get best model found by grid search
best_randomforest = grid_search.best_estimator_


# mean accuracy of the best model
winner_prediction = best_randomforest.predict(x_test_scaled)
randomforest_probability = best_randomforest.predict_proba(x_test_scaled)[:, 1]

#for standard deviation
best_index_randomForest = grid_search.best_index_

confusion_matrix(y_test, winner_prediction)


print(f"Best Parameters: {grid_search.best_params_}")

print("Random Forest Report")
print(classification_report(y_test, winner_prediction))

print(f"mean CV accuracy: {grid_search.best_score_:.4f}")
print(f"Train Accuracy: {best_randomforest.score(x_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {best_randomforest.score(x_test_scaled, y_test):.4f}")
print(f"Standard Deviation (Τυπική Απόκλιση): {grid_search.cv_results_['std_test_score'][best_index_randomForest]:.4f}")


print(f"ROC-AUC Score: {roc_auc_score(y_test, randomforest_probability):.4f}")

#LOGISTIC REGRESSION

logistic = LogisticRegression(solver='saga', max_iter=5000, random_state=42)

param_grid_logisticRegression = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'] 
}

grid_logisticRegression = GridSearchCV(logistic, param_grid_logisticRegression, cv=3, n_jobs=-1, verbose=1)
grid_logisticRegression.fit(x_train_scaled, y_train)

#find best model with grid search
best_logisticRegression = grid_logisticRegression.best_estimator_

winner_prediction = best_logisticRegression.predict(x_test_scaled)
logisticRegression_probability = best_logisticRegression.predict_proba(x_test_scaled)[:, 1]

best_index_logisticRegression = grid_logisticRegression.best_index_

confusion_matrix(y_test, winner_prediction)

print(f"Best Parameters: {grid_logisticRegression.best_params_}")
print("\nLogistic Regression Report")
print(classification_report(y_test, winner_prediction))

print(f"Mean CV Accuracy: {grid_logisticRegression.best_score_:.4f}")
print(f"Train Accuracy: {best_logisticRegression.score(x_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {best_logisticRegression.score(x_test_scaled, y_test):.4f}")
print(f"Standard Deviation (Τυπική Απόκλιση): {grid_logisticRegression.cv_results_['std_test_score'][best_index_logisticRegression]:.4f}")

print(f"ROC-AUC Score: {roc_auc_score(y_test, logisticRegression_probability):.4f}")

#SUPPORT VECTOR MACHINE

supportVectorMachine = SVC(probability=True,random_state=42)

param_grid_supportVectorMachine = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['rbf']
}

grid_supportVectorMachine = GridSearchCV(supportVectorMachine, param_grid_supportVectorMachine, cv=3, n_jobs=-1, verbose=1)
grid_supportVectorMachine.fit(x_train_scaled, y_train)

best_supportVectorMachine = grid_supportVectorMachine.best_estimator_

winner_prediction = best_supportVectorMachine.predict(x_test_scaled)

supportVectorMachine_probability = best_supportVectorMachine.predict_proba(x_test_scaled)[:, 1]

best_index_supportVectorMachine = grid_supportVectorMachine.best_index_

confusion_matrix(y_test, winner_prediction)

print(f"Best Parameters: {grid_supportVectorMachine.best_params_}")
print("\nSVM Report")
print(classification_report(y_test, winner_prediction))

print(f"Mean CV Accuracy: {grid_supportVectorMachine.best_score_:.4f}")
print(f"Train Accuracy: {best_supportVectorMachine.score(x_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {best_supportVectorMachine.score(x_test_scaled, y_test):.4f}")
print(f"Standard Deviation (Τυπική Απόκλιση): {grid_supportVectorMachine.cv_results_['std_test_score'][best_index_supportVectorMachine]:.4f}")

print(f"ROC-AUC Score: {roc_auc_score(y_test, supportVectorMachine_probability):.4f}")