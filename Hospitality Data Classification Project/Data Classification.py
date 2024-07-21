# -*- coding: utf-8 -*-
"""Untitled13.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VJJ0OgYOtSbLfdrF26GEq3ZFJXAkDT3j
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

# Load the datasets
train_data = pd.read_csv('/content/train.csv')
test_data = pd.read_csv('/content/test.csv')

# Exploratory Data Analysis (EDA)
def eda(data):
    print("First 5 rows of the dataset:")
    print(data.head())
    print("\nData Info:")
    print(data.info())
    print("\nSummary Statistics:")
    print(data.describe())
    print("\nMissing Values:")
    print(data.isnull().sum())
    sns.pairplot(data, hue='booking_status')
    plt.show()

eda(train_data)

# Handle missing values (if any)
imputer = SimpleImputer(strategy='most_frequent')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

# Feature Engineering: Creating additional features
train_data['total_nights'] = train_data['no_of_weekend_nights'] + train_data['no_of_week_nights']
train_data['guest_count'] = train_data['no_of_adults'] + train_data['no_of_children']
test_data['total_nights'] = test_data['no_of_weekend_nights'] + test_data['no_of_week_nights']
test_data['guest_count'] = test_data['no_of_adults'] + test_data['no_of_children']

# Define the feature columns and the target column
feature_columns = [
    'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
    'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 'lead_time',
    'arrival_year', 'arrival_month', 'arrival_date', 'market_segment_type', 'repeated_guest',
    'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
    'no_of_special_requests', 'total_nights', 'guest_count'
]
target_column = 'booking_status'

# Convert target column to numerical
label_encoder = LabelEncoder()
train_data[target_column] = label_encoder.fit_transform(train_data[target_column])

# Separate features and target variable from the training data
X_train = train_data[feature_columns]
y_train = train_data[target_column]

# Identify categorical and numerical columns
categorical_features = X_train.select_dtypes(include=['object']).columns
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Feature Selection
select_features = SelectKBest(chi2, k=10)

# Define models for comparison
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Hyperparameters for GridSearchCV and RandomizedSearchCV
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__solver': ['lbfgs', 'liblinear']
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30]
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }
}

# Perform hyperparameter tuning with RandomizedSearchCV
best_models = {}
for model_name, model in models.items():
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', select_features),
        ('classifier', model)
    ])

    param_grid = param_grids[model_name]
    grid_search = RandomizedSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, n_iter=10, random_state=42)
    grid_search.fit(X_train, y_train)

    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

# Ensemble Learning: Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('lr', best_models['Logistic Regression']),
    ('rf', best_models['Random Forest']),
    ('gb', best_models['Gradient Boosting'])
], voting='soft')

voting_clf.fit(X_train, y_train)

# Evaluate models on training data
for model_name, model in best_models.items():
    train_predictions = model.predict(X_train)
    accuracy = accuracy_score(y_train, train_predictions)
    precision = precision_score(y_train, train_predictions)
    recall = recall_score(y_train, train_predictions)
    f1 = f1_score(y_train, train_predictions)
    roc_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    print(f"\n{model_name} Model Performance:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")
    print("\nClassification Report:")
    print(classification_report(y_train, train_predictions))
    print("\nConfusion Matrix:")
    sns.heatmap(confusion_matrix(y_train, train_predictions), annot=True, fmt='d', cmap='Blues')
    plt.show()
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_train, model.predict_proba(X_train)[:, 1])
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='best')
    plt.show()

# Evaluate ensemble model on training data
train_predictions = voting_clf.predict(X_train)
accuracy = accuracy_score(y_train, train_predictions)
precision = precision_score(y_train, train_predictions)
recall = recall_score(y_train, train_predictions)
f1 = f1_score(y_train, train_predictions)
roc_auc = roc_auc_score(y_train, voting_clf.predict_proba(X_train)[:, 1])
print(f"\nVoting Classifier Model Performance:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
print("\nClassification Report:")
print(classification_report(y_train, train_predictions))
print("\nConfusion Matrix:")
sns.heatmap(confusion_matrix(y_train, train_predictions), annot=True, fmt='d', cmap='Blues')
plt.show()
# Plot ROC Curve for ensemble model
fpr, tpr, thresholds = roc_curve(y_train, voting_clf.predict_proba(X_train)[:, 1])
plt.plot(fpr, tpr, label=f"Voting Classifier (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Voting Classifier')
plt.legend(loc='best')
plt.show()

# Load test data for predictions
X_test = test_data[feature_columns]

# Predict on the test data using the best model
test_predictions = voting_clf.predict(X_test)
pd.Series(test_predictions).to_csv('pred.csv', index=False, header=False)

# Evaluate accuracy on training data and save to a.csv
accuracy = accuracy_score(y_train, train_predictions)
pd.Series([accuracy]).to_csv('a.csv', index=False, header=False)

# Classification report for precision and recall
report = classification_report(y_train, train_predictions, target_names=['Not_Canceled', 'Canceled'])
print(report)

# Performance analysis across classes
precision_not_canceled = precision_score(y_train, train_predictions, pos_label=0)
recall_not_canceled = recall_score(y_train, train_predictions, pos_label=0)
print(f"Precision for Not_Canceled: {precision_not_canceled}")
print(f"Recall for Not_Canceled: {recall_not_canceled}")

precision_canceled = precision_score(y_train, train_predictions, pos_label=1)
recall_canceled = recall_score(y_train, train_predictions, pos_label=1)
print(f"Precision for Canceled: {precision_canceled}")
print(f"Recall for Canceled: {recall_canceled}")