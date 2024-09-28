import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential

class OutputPredict:

    """Helper class containing functions for submitting the predictions."""

    @staticmethod
    def Output(model, assess_df: pd.DataFrame, assess_X: pd.DataFrame, file_path, threshold = None) -> None:
        """Formats the prediction and outputs to a csv.

        Args:
            model: A fitted Sklearn model. The model whose output we want to compute the predictions for.
            assess_df: The entire test set (Survey Test or Travel Test so that the ID's column can be extracted).
            assess_X_X: The scaled data for the test set.

        Returns:
            Outputs the predictions in csv format.

        """

        # make a prediction using the assessment set
        y: np.ndarray = model.predict(assess_X)

        if threshold:
            # only get the first column
            y: np.ndarray = y[:, 0]

            # convert to zeros and 1's
            y = y > threshold

        # convert to a dataframe
        y_df = pd.DataFrame({"ID": assess_df['ID'], "Overall_Experience": y})

        # get the IDs
        #y_df["ID"] = assess_df['ID']

        # reorder the columns
        #y_df = y_df[['ID', 'Overall_Experience']]

        # save to a csv
        y_df.to_csv(file_path, index=False)








