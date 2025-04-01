# CREDENTIAL RISK ASSESSMENT

1. Data Handling & Preprocessing
   
Loads the dataset (credit_risk_dataset.csv).

Performs exploratory analysis (.head(), .describe(), .info()).

Encodes categorical variables using one-hot encoding (pd.get_dummies()).

Generates a correlation matrix and visualizes it with seaborn.

2. Feature Engineering & Selection
   
Uses SelectKBest (f_classif) to choose the most relevant features.

Handles missing values using SimpleImputer.

3. Machine Learning Models
   
The notebook tests multiple models for risk classification:

Logistic Regression

Random Forest Classifier

Gradient Boosting & XGBoost

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Decision Tree & AdaBoost

It also applies SMOTE (Synthetic Minority Oversampling) to handle class imbalances.

4. Model Evaluation
   
Computes accuracy, precision, recall, ROC-AUC, confusion matrix.

Uses RandomizedSearchCV for hyperparameter tuning.

5. Visualization
   
Heatmaps for feature correlation.

Confusion matrices for model evaluation.
