import pandas as pd
import numpy as np
from math import log, fabs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class AdaBoostClassifier():
    def __init__(self, n, n_estimators=10):
        self.estimators = [DecisionTreeClassifier() for i in range(n_estimators)]
        self.sample_weights = [1 / n for i in range(n)]
        self.estimator_weights = [float(0) for i in range(n_estimators)]

    def fit(self, X, y):
        for i in range(len(self.estimators)):
            self.estimators[i].fit(X, y, self.sample_weights)
            weighted_error = self.weighted_error(self.estimators[i], X, y)
            if weighted_error == 0:
                weighted_error += 1e-10
            z = (1 - weighted_error) / weighted_error
            self.estimator_weights[i] = 0.5 * log(z)
            answers = self.estimators[i].predict(X)
            self.recompute_weights(self.estimator_weights[i], y, answers)
            self.normalize_weights()

    def weighted_error(self, estimator, X, y):
        error = 0
        sum = 0
        answers = estimator.predict(X)
        for i in range(len(answers)):
            sum += self.sample_weights[i]
            if answers[i] != y[i]:
                error += self.sample_weights[i]
        return error / sum

    def normalize_weights(self):
        sum = 0
        for weight in self.sample_weights:
            sum += weight
        for i in range(len(self.sample_weights)):
            self.sample_weights[i] = self.sample_weights[i] / sum

    def recompute_weights(self, estimator_weight, y, answers):
        for j in range(len(y)):
            if answers[j] == y[j]:
                self.sample_weights[j] *= np.exp(-estimator_weight)
            else:
                self.sample_weights[j] *= np.exp(estimator_weight)

    def predict(self, X):
        answers = []
        for i in range(len(X)):
            sum = 0
            for estimator, weight in zip(self.estimators, self.estimator_weights):
                x_i = np.ndarray(shape=(1, 18), buffer=np.array(X[i]))
                if weight != 0:
                    sum += weight * int(estimator.predict(x_i)[0])
            if sum >= 0:
                answers.append(1)
            else:
                answers.append(-1)
        return np.array(answers)


def simplify(data):
    for i in range(len(data)):
        if data[i] == 'EUROPE':
            data[i] = 1
        else:
            data[i] = -1


def find_median(s):
    median = 0
    i = int(len(s) / 2)
    if len(s) % 2 == 0:
        median = s[i][0] + s[i + 1][0]
    else:
        median = s[i + 1][0]
    return median, i


def without_outliers(X, y, sample_weights):
    X_new = []
    y_new = []
    s = []
    j = 0
    for i in sample_weights:
        s.append((i, j))
        j += 1
    s.sort(key=lambda k: k[0])
    q1, i = find_median(s)
    q2 = find_median(s[:i + 1])[0]
    q3 = find_median(s[i + 1:])[0]
    diff = q3 - q2
    b = diff * 1.5
    lower = q2 - b
    higher = q3 + b
    for i in range(len(s)):
        if s[i][0] >= lower and s[i][0] <= higher:
            X_new.append(X[s[i][1]])
            y_new.append(y[s[i][1]])
    X_new = np.array(X_new)
    y_new = np.array(y_new)
    return X_new, y_new


data = pd.read_csv("countries.csv")
data = data.fillna(0)

X = np.array(data.drop(['Region', 'Country'], axis=1))
y = np.array(data.Region)
simplify(y)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y)

adaboost = AdaBoostClassifier(len(X_train))

adaboost.fit(X_train, y_train)

y_pred = adaboost.predict(X_test)

print('Accuracy score with outliers: ')
print(accuracy_score(y_test, y_pred))

X_train, y_train = without_outliers(X_train, y_train, adaboost.sample_weights)

adaboost.fit(X_test, y_test)

y_pred = adaboost.predict(X_test)

print('Accuracy score without outliers: ')
print(accuracy_score(y_test, y_pred))
