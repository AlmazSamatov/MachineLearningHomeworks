import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def simplify(data):
    for i in range(len(data)):
        if data[i] == 'EUROPE':
            data[i] = 1
        else:
            data[i] = -1


def normalize_weights(sample_weights):
    sum = 0
    for weight in sample_weights:
        sum += weight
    for i in range(len(sample_weights)):
        sample_weights[i] = sample_weights[i] / sum
    return sample_weights


def recompute_weights(sample_weights, estimator_weight, y, answers):
    for j in range(len(y)):
        if answers[j] == y[j]:
            sample_weights[j] *= np.exp(-estimator_weight)
        else:
            sample_weights[j] *= np.exp(estimator_weight)
    return sample_weights


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

adaboost = AdaBoostClassifier(algorithm='SAMME')

adaboost.fit(X_train, y_train)

f = adaboost.estimators_
w = adaboost.estimator_weights_

sample_weights = [1 / len(X_train) for i in range(len(X_train))]

for i in range(adaboost.n_estimators):
    answer = f[i].predict(X_train)
    sample_weights = recompute_weights(sample_weights, w[i], y_train, answer)
    sample_weights = normalize_weights(sample_weights)

y_pred = adaboost.predict(X_test)

print('Accuracy score with outliers: ')
print(accuracy_score(y_test, y_pred))

X_train, y_train = without_outliers(X_train, y_train, sample_weights)

adaboost.fit(X_train, y_train)

y_pred = adaboost.predict(X_test)

print('Accuracy score without outliers: ')
print(accuracy_score(y_test, y_pred))
