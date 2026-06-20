# Chess Game Winner Prediction

This repository contains a machine learning pipeline written in Python to predict the outcome of a chess match (White wins vs. Black wins) using data from played games. It evaluates and compares three popular classification algorithms: 
**1. Random Forest** 
**2. Logistic Regression** 
**3. Support Vector Machines (SVM)**

## Dataset & Feature Engineering
The pipeline utilizes chess game metadata (`games.csv`).

### Selected Features Matrix
The model trains on the following final feature subset:
* `white_rating`, `black_rating`, `rating_diff` (Player Skill Data)
* `rated` (Whether the match affects official rankings)
* `base_value`, `increment_value` (Time Control Controls)
* `opening_eco`, `opening_ply` (Opening Theory Data)

Target Variable ($y$)
* **`winner`**: Binary classification target mapped as:
    * `0`: White Wins
    * `1`: Black Wins
 
#### Data Preprocessing

To ensure optimal performance and prevent data leakage, the pipeline applies several preprocessing steps:

1.  **Filtering**: All drawn games are dropped from the dataset.
2.  **Missing Values**: Handled using explicit list-wise deletion (`dropna()`).
3.  **Categorical Encoding**: 
    * Features like `rated` and `opening_eco` are processed using `LabelEncoder`.
    * The target variable `winner` is explicitly mapped to binary numeric flags.
4.  **Dataset Splitting**: The dataset is split into an **80% Training set** and a **20% Testing set**, stratified according to the target variable to ensure proportional class distribution.
5.  **Feature Scaling**: Numeric columns are normalized using `StandardScaler` fitted *only* on the training data.

##### Installation & Usage
1. **Navigate to the project root folder:**
    ```bash
       cd Chess_Winner_Predictor
    ```
   **Download python packages:**
     ```bash
       pip install -r requirements.txt
     ```
3. **Execute the script:**
     ```bash
       python3 model.py
     ```
