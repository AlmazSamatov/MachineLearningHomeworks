import pandas as pd
from matplotlib import pyplot as plt

customers = pd.read_csv("mall_customers.csv")
genre = {'Male': 'blue', 'Female':'red'}

plt.scatter(customers.Income, customers.Spending, c=customers.Genre.apply(lambda x: genre[x]))
plt.xlabel("income")
plt.ylabel("spending")
plt.legend(genre)
plt.show()