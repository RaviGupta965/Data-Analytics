import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

def split_dataset(X, y, test_size, random_state):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def scale_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.n_iterations):
            model = self.sigmoid(np.dot(X, self.weights) + self.bias)
            error = model - y
            dw = np.dot(X.T, error) / len(X)
            db = np.sum(error) / len(X)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return model.round()

def confusion_matrix(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    cm = np.zeros((2, 2))
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    return cm

def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def visualize_classification(X_set, y_set, classifier, title, mean, std):
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.1))
    Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape)
    plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))
    ticks_x = np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 1)
    ticks_y = np.arange(X_set[:, 1].min(), X_set[:, 1].max() + 1, 1)
    plt.xticks(ticks_x, (ticks_x * std[0] + mean[0]).astype(int))
    plt.yticks(ticks_y, (ticks_y * std[1] + mean[1]).astype(int))
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=np.array([ListedColormap(('red', 'green'))(i)]), label=j)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.25, random_state=0)

# Feature Scaling
X_train, mean_train, std_train = scale_features(X_train)
X_test = (X_test - mean_train) / std_train

# Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Confusion Matrix and Accuracy Score
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n', cm)
print('Accuracy Score:', accuracy_score(y_test, y_pred))

# Visualising the Training set results
visualize_classification(X_train, y_train, classifier, 'Logistic Regression (Training set)', mean_train, std_train)

# Visualising the Test set results
visualize_classification(X_test, y_test, classifier, 'Logistic Regression (Test set)', mean_train, std_train)
