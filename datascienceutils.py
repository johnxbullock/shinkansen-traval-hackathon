import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataScienceUtils:
    @staticmethod
    def inspect_variable_values(df: pd.DataFrame):

        # iterate through each column
        for i, col in enumerate(df.columns):

            # get a pandas series of the values in each column
            vals: np.ndarray = df[col].unique()

            # get the data type of the variables
            data_type = vals.dtype

            non_null_pct = df[col].isnull().sum() / df[col].count() * 100

            # print the column name
            print(f'{i + 1}. {col} \n')

            # print the columns data type
            print(f'Data type: {data_type}')

            # print the number of values the variable takes on
            print("Number of distinct values:", len(vals))

            print(f"Percentage of entries null: {non_null_pct}")

            # print the values inside the column

            # if the variable takes on < 20 vals, print all the values
            if len(vals) < 20:
                print("Values taken on by the variable:", vals, '\n')
            else:
                # otherwise, if the variable takes on > 20 values and is numeric, then print the range of values the variable takes on
                if isinstance(data_type, (np.dtypes.Int64DType, np.dtypes.Float64DType)):
                    print("Values taken on by the variable:", f"Range: {vals.min()} - {vals.max()}", '\n')
                else:
                    # otherwise the values are categorical, then print only the first 20 values
                    print("Values taken on by the variable:", vals[:20], '\n')

            # new line between columns
            print()

    @staticmethod
    def get_vars_by_type(df: pd.DataFrame) -> tuple[list, list, list]:
        """
        Args:
            df: The dataframe whose variables we would like to categorize into numerical, binary and categorical.
        Returns:
            A tuple of lists, elem 1: Numerical, elem 2: Categorical, elem 3: Binary
        """
        # list of column names for numerical variables
        numerical_vars: list = []

        # list of column names for categorical variables
        categorical_vars: list = []

        # list of column names for binary variables
        binary_vars: list = []

        binary_words_remove = ["Unknown", "other", "unknown", "Other", np.nan]

        # loop through the data and add the column name to the appropriate list
        for col in df:

            vals: np.ndarray = df[col].dropna(inplace=False).unique()

            # remove any extra words
            vals = vals[~np.isin(vals, binary_words_remove)]

            data_type = vals.dtype

            # first test whether the data is numeric
            if isinstance(data_type, np.dtypes.Float64DType):
                numerical_vars.append(col)
            elif isinstance(data_type, (np.dtypes.Int64DType, int)):
                if (vals > 1).any() or (vals < 0).any():
                    numerical_vars.append(col)
                else:
                    binary_vars.append(col)
            elif np.isin(["Yes", "yes", "no", "NO"], vals).any() or len(vals) == 2:
                binary_vars.append(col)
            else:
                categorical_vars.append(col)

        return numerical_vars, categorical_vars, binary_vars

    @staticmethod
    def analyze_categorical_features(df: pd.DataFrame, cat_features: list, bivariate=False, target=None):

        if not bivariate:
            # loop through each column and create a countplot
            for i, col in enumerate(cat_features):
                print(f"{i + 1}. Analysis for {col}:", '\n')
                plt.figure(figsize=(10, 5))
                sns.countplot(x=df[col])
                plt.title(f"{col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.show()
                print(df[col].value_counts(normalize=True), "\n")

        else:
            # compute the joint distribution of the status, and each of the variables in cat_features
            joint_dist = pd.pivot_table(df, index=target, columns=cat_features, aggfunc='size')

            # drop the na columns
            joint_dist = joint_dist.loc[1, :].dropna()

            # sort values in descending order
            joint_dist.sort_values(ascending=False, inplace=True)

            # make a list of observed values
            observed_values: list = [[value[i] for value in joint_dist.index.values] for i in range(0, len(cat_features))]

            # make a list of pandas series using the list of observed values
            observed_values_series: list = [pd.Series(observed_values[i]).value_counts(normalize=True) for i in
                                            range(0, len(cat_features))]

            for i, col in enumerate(cat_features):
                # compute a pivot table with values being the counts of each observation
                table = pd.pivot_table(df, index=target, columns=col, aggfunc='size')

                # compute the proportion rather than count
                table = table.div(table.sum(axis=0), axis=1)

                # split words in the title to capitalize
                title = col.split("_")

                # capitalize each word in the title and join into a single word
                title = ' '.join(word.capitalize() for word in title)

                # compute the proportions of each value of the variable falling into 1 or 0 of the status variable
                proportions: pd.DataFrame = pd.crosstab(df[col], df[target], normalize=True).reset_index()

                # melt for plotting
                proportions: pd.DataFrame = proportions.melt(id_vars=col, var_name=target, value_name='Proportion')

                print(f"Bivariate EDA for {col}: ", '\n')

                # initialize the figure
                plt.figure(figsize=(10, 3))

                # add a title
                plt.title(f"Bar plot of {title} by {target}")

                # plot the barplot
                sns.barplot(x=col, y="Proportion", hue=df[target].astype(str), data=proportions)

                plt.show()
    @staticmethod
    def analyze_numerical_features(df: pd.DataFrame, numerical_vars: list, bivariate=False, target=None):
        # view summary statistics of numerical variables
        if not bivariate:
            summary_data = df[numerical_vars].describe().T
        else:
            summary_data: list = [df.loc[df[target] == i, numerical_vars].describe() for i in range(2)]

        # loop through the columns of the dataframe
        for col in numerical_vars:

            # initialize a figure
            fig = plt.figure(figsize=(10, 3))

            plt.suptitle(f"{col}")

            # add subplots to the figure
            ax_1 = fig.add_subplot(1, 2, 1)
            ax_2 = fig.add_subplot(1, 2, 2)

            if bivariate:

                # create the histogram and add it to the first ax
                sns.histplot(data=df, x=col, ax=ax_1, kde=True, stat='proportion', hue=target)

                # create the boxplot and add it to the second ax
                sns.boxplot(data=df, x=col, ax=ax_2, showmeans=True, hue=target)

                # display the plot
                plt.show()

                # concatenate the summary stats for positive and negative cases
                comparison = pd.concat([summary_data[0][col], summary_data[1][col]], axis=1)

                # rename the columns of the df
                comparison.columns = [f"{col}_0", f"{col}_1"]

                print(
                    f"Comparison of the summary statistics for paying and non-paying customers for the {col} variable",
                    "\n")
                print(comparison, "\n")

            else:

                # view summary statistics of numerical variables
                summary_data = df[numerical_vars].describe().T
                # create the histogram and add it to the first ax
                sns.histplot(data=df, x=col, ax=ax_1, kde=True, stat='proportion')

                # create the boxplot and add it to the second ax
                sns.boxplot(data=df, x=col, ax=ax_2, showmeans=True)

                # display the plot
                plt.show()

                print(f"Summary statistics for the {col} variable", "\n")
                print(summary_data.loc[col, :], "\n")

    @staticmethod
    def analyze_binary_features(df: pd.DataFrame, binary_vars: list):
        # loop through the columns of the dataframe
        for i, col in enumerate(binary_vars):
            print(f"{i}. Analysis for {col}:")

            plt.figure(figsize=(10, 3))

            plt.suptitle(f"{col} normalized bar plot")

            # use value counts to get the proportions of Yes and No
            bin_vals: pd.Series = df[col].value_counts(normalize=True)

            # create a dataframe using the proportions
            bin_df = pd.DataFrame({col: bin_vals.index, 'Proportion': bin_vals.values})

            sns.barplot(data=bin_df, x=col, y='Proportion')

            plt.show()

            print(df[col].value_counts(normalize=True))

    @staticmethod
    def evaluate_model():
        pass

    @staticmethod
    def hello():
        print("Hello World!")
