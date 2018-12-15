import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, decomposition

# generate data

iris = datasets.load_iris()

x = iris.data
y = iris.target

# from scratch

### 2 CENTER DATA
x_centered = x - x.mean()

### 3 PROJECT DATA
covariance_matrix = np.cov(x_centered.T)
### next step you need to find eigenvalues and eigenvectors of covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
### find out which eigenvectors you should choose based on eigenvalues
projected_data = np.dot(x_centered, eigenvectors[:,[0,1]])

### 4 RESTORE DATA

restored_data = np.dot(projected_data, eigenvectors[:,[0,1]].T) + x.mean()

# plot data

plt.plot(projected_data[y == 0, 0], projected_data[y == 0, 1], 'bo', label='Setosa')
plt.plot(projected_data[y == 1, 0], projected_data[y == 1, 1], 'go', label='Versicolour')
plt.plot(projected_data[y == 2, 0], projected_data[y == 2, 1], 'ro', label='Virginica')
plt.legend(loc=0)
plt.show()

# sklearn method

pca = decomposition.PCA(n_components=2)
pca.fit(x)
X_pca = pca.transform(x)

plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
plt.legend(loc=0)
plt.show()



