import pandas as pd
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error


def ridge_cv(X, y, alphas):
    '''
    performs cross-validation of Ridge regression to find optimal value of alpha
    Arguments:
    X - training data
    Y - training data labels
    aplhas - list of alphas to choose from

    Returns:
    results - list of mse (mean squarred error), for each of possible alphas
    '''
    length = len(X)
    splits = leave_one_out_split(length)
    results = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, normalize=True)
        mse = 0
        for split in splits:
            index_train, index_test = split[0], split[1]
            # (a) split the data into test and train as per the split indices
            # (b) fit the model
            # (c) find mse - mean squared error

            # split data
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            for i in range(length):
                if i in index_train:
                    x_train.append(X.values[i])
                    y_train.append(y.values[i])
                if i in index_test:
                    x_test.append(X.values[i])
                    y_test.append(y.values[i])

            # fit the model
            model.fit(x_train, y_train)
            # predict
            y_predicted = model.predict(x_test)
            # calculate mse
            mse += mean_squared_error(y_test, y_predicted)
        results.append(mse / length)
    return results


def leave_one_out_split(length):
    '''
    the method should perform splits according to leave-one-out cross-validation, i.e.:
    each time only one sample is used for testing, all others are used for training

    returns a list of tuples of train and test indexes for each split:
    [([train_indices_1], [test_index_1]), ([train_indices_2], [test_index_2]), ...]
    each tuple is a split

    pay attention - we don't split actual data, we only generate indices for splitting

    Arguments:
    length - #rows in dataset

    Returns:
    splits - list of tuples
    '''
    splits = []
    for i in range(length):
        test_indices = [i]
        train_indices = []
        for j in range(length):
            if i != j:
                train_indices.append(j)
        splits.append((train_indices, test_indices))
    return splits


# loading and pre-processing the dataset
hitters = pd.read_csv("Hitters.csv").dropna().drop("Player", axis=1)
dummies = pd.get_dummies(hitters[['League', 'Division', 'NewLeague']])

# Dropping the column with the independent variable (Salary), and columns for which we created dummy variables
X_ = hitters.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# Defining the feature set X.
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
y = hitters.Salary

alphas = [1e-15, 1e-10, 1e-5, 1e-3, 1e-2, 1, 3, 5]

results = ridge_cv(X, y, alphas)
