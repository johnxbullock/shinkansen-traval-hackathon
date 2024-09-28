import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# class to evaluate various models
class Evaluator:

    @staticmethod
    # Creating metric function
    def sklearn_evaluation(x_test, y_true, model=None):

        y_pred = model.predict(x_test)

        print(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 5))

        sns.heatmap(cm, annot=True, fmt='.2f')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
