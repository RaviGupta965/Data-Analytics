import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def split_data(X, y, test_size, random_state):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def linear_regression(X, y):
    n = np.size(X)
    mean_x, mean_y = np.mean(X), np.mean(y)
    SS_xy = np.sum(y * X) - n * mean_y * mean_x
    SS_xx = np.sum(X * X) - n * mean_x * mean_x
    b_1 = SS_xy / SS_xx
    b_0 = mean_y - b_1 * mean_x
    return b_0, b_1


def multiple_linear_regression(X, y):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return weights


def predict(X, b_0, b_1):
    return b_0 + b_1 * X


def predict_multiple(X, weights):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X.dot(weights)


dataset = pd.read_csv('Salary Data.csv')
dataset = dataset.dropna()


print(dataset.head())


X = dataset.iloc[:, -2].values
y = dataset.iloc[:, -1].values


X_train, X_test, y_train, y_test = split_data(X, y, 1/3, 0)
b_0, b_1 = linear_regression(X_train.flatten(), y_train)


y_pred = predict(X_test.flatten(), b_0, b_1)
mean_y = np.mean(y_test)
SS_res = np.sum((y_test - y_pred) ** 2)
SS_tot = np.sum((y_test - mean_y) ** 2)


r_squared = 1 - SS_res / SS_tot
print(f'R-Squared: {r_squared}')


plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, predict(X_train.flatten(), b_0, b_1), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, predict(X_train.flatten(), b_0, b_1), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
