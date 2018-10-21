import pandas as pd

iris = pd.read_csv("iris.csv")

iris = iris.sample(frac=1).reset_index(drop=True)

training, test = iris[:110], iris[110:]

training.to_csv("training_data.csv")
test.to_csv("test_data.csv")
