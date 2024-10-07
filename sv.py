# import the necessary libraries

# numpy for numerical computations and arrays
import numpy as np
import pandas as pd

# for splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# import the Support Vector Module
from sklearn.svm import SVC

# evaluation metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, recall_score, average_precision_score
from sklearn.metrics import roc_curve

# for tuning the model
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, RandomizedSearchCV

# import libraries for visualizing results
from matplotlib import pyplot as plt
import seaborn as sns

# import utility functions
#from Utils.evaluate import Evaluator

from scipy.stats import uniform

# use the scaled data
df = pd.read_csv('ProcessedData/train_scaled.csv')

# get the features for the assessment dt
df_test: pd.DataFrame = pd.read_csv('ProcessedData/X_test_scaled.csv')

# get the ID's to add to the test data
IDs: pd.DataFrame = pd.read_csv('Data/Surveydata_test.csv')['ID']

# get the assessment data set
df_assess = pd.read_csv('Data/Surveydata_test.csv')

# import the assessment data
X_assess_df = pd.read_csv('ProcessedData/X_test_scaled.csv')

# inspect the dataset
df.head()

# split data into features and target

# class
Y = df['Overall_Experience']

# features
X = df.drop(columns='Overall_Experience')

# split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# Creating metric function
def metrics_score(model, x_test, y_test):
    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 5))

    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Attrite', 'Attrite'],
                yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Initialize the SVM model
svm_model = SVC()

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'C': uniform(0.1, 10),  # Sampling C values from a uniform distribution between 0.1 and 10
    'gamma': uniform(0.001, 1),  # Sampling gamma values from a uniform distribution between 0.001 and 1
    'kernel': ['rbf', 'poly'],  # Different kernels to try
    'degree': [2, 3, 4]  # Only relevant if 'poly' is selected as kernel
}

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=svm_model,  # The SVM model
    param_distributions=param_dist,  # Parameter grid to sample from
    n_iter=50,  # Number of different combinations to try
    scoring='accuracy',  # Evaluation metric
    cv=5,  # Cross-validation folds
    verbose=1,  # Display information during fitting
    random_state=42,  # For reproducibility
    n_jobs=-1  # Use all available cores for parallel processing
)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, Y_train)

# Display the best parameters and best score found
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_}")

# Evaluate the best model on the test set
best_svm_model = random_search.best_estimator_
test_score = best_svm_model.score(X_test, Y_test)
print(f"Test Set Score: {test_score}")

# make predictions on the test data
y: np.ndarray = best_svm_model.predict(df_test)


print('Converting to dataframe.')
# convert to a dataframe
y_df = pd.DataFrame({'ID': IDs, "Overall_Experience": y})

print('Saving as csv.')
# print for submission
y_df.to_csv(f'Predictions/svm1.csv', index=False)

print('Done.')
