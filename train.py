#!/usr/bin/env python
# coding: utf-8




# load data
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import pickle




data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

data['Sleep Disorder'] = data['Sleep Disorder'].fillna('Not Present')

print(data.head(10))

#list the variables

data.dtypes

"""# Exploratory data analysis"""

data.shape

data.duplicated().sum()

data.head().T

data.columns

for col in data.columns:
  print(col)
  print(data[col].unique()[:10])
  print(data[col].nunique())
  print()

data.describe()

data.isnull().sum()

data.duplicated().sum()

data["Sleep Disorder"].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt

pd.crosstab(data["Gender"],data["Occupation"]).plot(kind="barh")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
plt.show()

pd.crosstab(data["Gender"],data["Sleep Disorder"]).plot(kind="bar")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
plt.show()

pd.crosstab(data["Gender"],data["Sleep Disorder"]).plot(kind="bar")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
plt.show()

pd.crosstab(data["Occupation"],data["Sleep Disorder"]).plot(kind="pie", subplots=True, figsize=(20, 20))
plt.show()

# Applying Label Encoding

from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
le = LabelEncoder()

# Label Encoding
data['Blood Pressure'] = le.fit_transform(data['Blood Pressure'])
data['Gender'] = le.fit_transform(data['Gender'])
data['Occupation'] = le.fit_transform(data['Occupation'])
data['BMI Category'] = le.fit_transform(data['BMI Category'])
data['Sleep Disorder'] = le.fit_transform(data['Sleep Disorder'])


print(data['Blood Pressure'].head())

data.info()

data.head(5)

data.columns

data.shape

data.isnull().sum()

data["Sleep Disorder"].value_counts()

"""# Feature importance analysis#



"""

correlation_matrix = data.corr(method='pearson')

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()




X = data.drop(['Person ID', 'Sleep Disorder'], axis=1)
y = data['Sleep Disorder']

# Calculation for mutual information
mi_scores = mutual_info_classif(X, y)

# dataframe of features and their mutual information scores
mi_data = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
mi_data = mi_data.sort_values('mi_score', ascending=False).reset_index(drop=True)


print(mi_data)

# Visualize the top 15 features
plt.figure(figsize=(10, 6))
plt.bar(mi_data['feature'][:15], mi_data['mi_score'][:15])
plt.title('Top 15 Features by Mutual Information')
plt.xlabel('Features')
plt.ylabel('Mutual Information Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print(X.columns)

print(X.shape)

print(y)

"""# Model selection and parameter tuning"""

# Stratify the data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=55)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=55)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y)

# LogisticRegression




# A simple parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 2, 50, 100],
    'max_iter': [500, 1000]
}

# The logistic regression model
lr = LogisticRegression(random_state=55)

# Grid search
grid_search = GridSearchCV(
    lr,
    param_grid,
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best estimator
lr_best = grid_search.best_estimator_

# Best parameters
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation balanced accuracy score:", grid_search.best_score_)

# Evaluation on validation set
val_pred = lr_best.predict(X_val)
val_bal = balanced_accuracy_score(y_val, val_pred)
val_precision = precision_score(y_val, val_pred, average='weighted')
val_recall = recall_score(y_val, val_pred, average='weighted')
val_f1 = f1_score(y_val, val_pred, average='weighted')


print("Validation balanced accuracy:", val_bal)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1 Score:", val_f1)

# Evaluation on test set
test_pred = lr_best.predict(X_test)
test_bal = balanced_accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred, average='weighted')
test_recall = recall_score(y_test, test_pred, average='weighted')
test_f1 = f1_score(y_test, test_pred, average='weighted')

print("Test balanced accuracy:", test_bal)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)



# RandomForest


param_grid = {
    'n_estimators': [100, 150, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}


rf = RandomForestClassifier(random_state=55)


grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1
)


grid_search.fit(X_train, y_train)


rf_best = grid_search.best_estimator_


print("Best parameters:", grid_search.best_params_)
print("Best cross-validation balanced accuracy score:", grid_search.best_score_)


val_pred = rf_best.predict(X_val)
val_bal = balanced_accuracy_score(y_val, val_pred)
val_precision = precision_score(y_val, val_pred, average='weighted')
val_recall = recall_score(y_val, val_pred, average='weighted')
val_f1 = f1_score(y_val, val_pred, average='weighted')

print("Validation balanced accuracy:", val_bal)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1 Score:", val_f1)


test_pred = rf_best.predict(X_test)
test_bal = balanced_accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred, average='weighted')
test_recall = recall_score(y_test, test_pred, average='weighted')
test_f1 = f1_score(y_test, test_pred, average='weighted')

print("Test balanced accuracy:", test_bal)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)



#Gradient Boosting model


param_grid = {
    'n_estimators': [100, 150, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}


gb = GradientBoostingClassifier(random_state=55)


grid_search = GridSearchCV(
    gb,
    param_grid,
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1
)


grid_search.fit(X_train, y_train)


gb_best = grid_search.best_estimator_


print("Best parameters:", grid_search.best_params_)
print("Best cross-validation balanced accuracy score:", grid_search.best_score_)


val_pred = gb_best.predict(X_val)
val_bal = balanced_accuracy_score(y_val, val_pred)
val_precision = precision_score(y_val, val_pred, average='weighted')
val_recall = recall_score(y_val, val_pred, average='weighted')
val_f1 = f1_score(y_val, val_pred, average='weighted')

print("Validation balanced accuracy:", val_bal)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1 Score:", val_f1)


test_pred = gb_best.predict(X_test)
test_bal = balanced_accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred, average='weighted')
test_recall = recall_score(y_test, test_pred, average='weighted')
test_f1 = f1_score(y_test, test_pred, average='weighted')

print("Test balanced accuracy:", test_bal)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)

"""The Random Forest model emerges as the best-performing model among the three evaluated, demonstrating superior metrics across both validation and test sets. With a validation balanced accuracy of (0.8674) and a test balanced accuracy of (0.9266), it significantly outperforms both Logistic Regression and Gradient Boosting in terms of precision (0.9331), recall (0.9200), and F1 score (0.9233)

# Save the model
"""



filename = 'random_forest_model.pkl'
pickle.dump(rf_best, open(filename, 'wb'))
print(f"Model saved as {filename}")

