import numpy as np
import pandas as pd

def partition_discrete(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}


def select_value(y, x, attribute):
    needed_x = x[:, attribute]
    needed_x.sort()
    new_x = []

    for i in range(len(needed_x) - 1):
        new_x.append((needed_x[i] + needed_x[i + 1]) / 2)

    max_entropy = -1 * float("inf")
    best_value = -1

    for i in range(len(new_x)):
        lower = []
        bigger = []
        for row in needed_x:
            if row > new_x[i]:
                bigger.append(row)
            else:
                lower.append(row)
        gain = information_gain_for_real_values(y, [lower, bigger])
        if max_entropy < gain:
            best_value = new_x[i]
            max_entropy = gain

    return best_value


def partition_continuous(y, x, attribute):
    best_value = select_value(y, x, attribute)
    lower_or_equal = []
    bigger = []
    values = x[:, attribute]
    for i in range(len(x)):
        if values[i] <= best_value:
            lower_or_equal.append(i)
        else:
            bigger.append(i)
    return {'<= ' + str(best_value): lower_or_equal, '> ' + str(best_value): bigger}


def information_gain_for_real_values(union, subsets):
    before_split = entropy(union)

    w = [len(subset) / len(union) for subset in subsets]

    after_split = 0

    for i in range(len(subsets)):
        after_split += w[i] * entropy(subsets[i])

    gain = before_split - after_split

    return gain


def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def information_gain(y, x):
    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float') / len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res


def information_gain_ratio(y, x):
    if entropy(x) == 0:
        return 0
    else:
        return information_gain(y, x) / entropy(x)


def is_pure(s):
    return len(set(s)) == 1


def recursive_split(x, y, fields):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) <= 5:
        return y

    # We get attribute that gives the highest information gain
    gain = np.array([information_gain_ratio(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y

    if is_continuous(selected_attr):
        # if it is continuous
        sets = partition_continuous(y, x, selected_attr)

        res = {}

        for k, v in sets.items():
            y_subset = y.take(v, axis=0)
            x_subset = x.take(v, axis=0)

            res["{} {}".format(fields[selected_attr], k)] = recursive_split(x_subset, y_subset, fields)

    else:
        # if it is discrete
        # We split using the selected attribute
        sets = partition_discrete(x[:, selected_attr])

        res = {}

        for k, v in sets.items():
            y_subset = y.take(v, axis=0)
            x_subset = x.take(v, axis=0)

            res["{} = {}".format(fields[selected_attr], k)] = recursive_split(x_subset, y_subset, fields)

    return res


def predict(tree, fields, x_test):
    y_test = []
    for i in range(len(x_test)):
        temp_tree = tree
        while not isinstance(temp_tree, np.ndarray):
            for k, v in temp_tree.items():

                end = [k.find('<='), k.find('>'), k.find('=')]
                end_index = -1
                for j in end:
                    if j != -1:
                        end_index = j

                attr = k[:end_index - 1].strip()
                value = k[end_index + 2:].strip()
                index = fields.index(attr)

                if end[0] != -1 and x_test[i][index] <= float(value):
                    temp_tree = v
                if end[1] != -1 and x_test[i][index] > float(value):
                    temp_tree = v
                elif x_test[i][index] == value or (
                        isinstance(x_test[i][index], int) and x_test[i][index] == int(value)):
                    temp_tree = v
        y_test.append(temp_tree[0])
    return y_test


def is_continuous(field_index):
    global X
    val, counts = np.unique(X[:, field_index], return_counts=True)
    return len(val) > 5 and isinstance(val[0], float)


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

train_set = pd.read_csv('titanic_modified.csv')

train_set = train_set.dropna()

X = train_set.iloc[:, :6].values
y = train_set.iloc[:, 6].values

fields = list(train_set.columns.values)
tree = recursive_split(X, y, fields)
print(tree)
