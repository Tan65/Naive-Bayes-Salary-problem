# Naive-Bayes-Salary-problem
Prepare a classification model using Naive Bayes  for salary data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Salary Train and Salary Test data
salary_train = pd.read_csv('SalaryData_Train.csv')
salary_test = pd.read_csv('SalaryData_Test.csv')

# Concatenate train and test data for preprocessing
data = pd.concat([salary_train, salary_test])

# Handling missing values
data = data.dropna()

# Convert categorical variables using LabelEncoder
categorical_cols = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Split the data back into train and test
train_data = data[:salary_train.shape[0]]
test_data = data[salary_train.shape[0]:]

# Split the data into features and target
X_train = train_data.drop('Salary', axis=1)
y_train = train_data['Salary']
X_test = test_data.drop('Salary', axis=1)
y_test = test_data['Salary']

# Scale numerical features using StandardScaler
numerical_cols = ['age', 'educationno', 'capitalgain', 'capitalloss', 'hoursperweek']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Ensure all values in X_train and X_test are non-negative
X_train = X_train.abs()
X_test = X_test.abs()

# Feature selection using SelectKBest and chi2
selector = SelectKBest(chi2, k=8)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Convert selected features back to DataFrame
selected_indices = selector.get_support(indices=True)
selected_features = X_train.columns[selected_indices]
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

# Create a Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Define hyperparameters for tuning
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7]
}

# Perform grid search cross-validation
grid_search = GridSearchCV(nb_classifier, param_grid, cv=5)
grid_search.fit(X_train_selected_df, y_train)

# Retrieve the best classifier
best_classifier = grid_search.best_estimator_

# Predict the target values for the test data
y_pred = best_classifier.predict(X_test_selected_df)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualizations

# Univariate Analysis - Histograms
plt.figure(figsize=(15, 10))
for i, col in enumerate(X_train_selected_df.columns):
    plt.subplot(3, 3, i+1)
    sns.histplot(X_train_selected_df[col], kde=True)
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

# Bivariate Analysis - Pairplot
pairplot_data = pd.concat([X_train_selected_df, y_train], axis=1)
plt.figure(figsize=(15, 10))
sns.pairplot(pairplot_data, hue='Salary', diag_kind='kde')
plt.show()

# Multivariate Analysis - Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = pairplot_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
