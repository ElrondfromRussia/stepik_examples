from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model

import numpy as np



# ########################################################
# ########################################################
class LinearRegression:

    def __init__(self, mae_metric=False):
        """
            @param mae_metrics: В случае True необходимо следить за
            метрикой MAE во время обучения, иначе - за метрикой MSE
        """
        self.metric = self.calc_mse_metric if not mae_metric else self.calc_mae_metric

    def calc_mae_metric(self, preds, y):
        """
            @param preds: предсказания модели
            @param y: истиные значения
            @return mae: значение MAE
        """
        return np.sum(np.abs(y - preds)) / y.shape[0]

    def calc_mse_metric(self, preds, y):
        """
            @param preds: предсказания модели
            @param y: истиные значения
            @return mse: значение MSE
        """
        return np.sum((y - preds) ** 2) / y.shape[0]

    def init_weights(self, input_size, output_size):
        """
            Инициализирует параметры модели
            W - матрица размерности (input_size, output_size)
            инициализируется рандомными числами из
            нормального распределения (np.random.normal)
            со средним 0 и стандартным отклонением 0.01
            b - вектор размерности (1, output_size)
            инициализируется нулями
        """
        np.random.seed(42)
        self.W = np.random.normal(0, 0.01, size=(input_size, output_size))
        self.b = np.zeros((1, output_size), dtype=float)

    def fit(self, X, y, num_epochs=1000, lr=0.001):
        """
            Обучение модели линейной регрессии методом градиентного спуска
            @param X: размерности (num_samples, input_shape)
            @param y: размерности (num_samples, output_shape)
            @param num_epochs: количество итераций градиентного спуска
            @param lr: шаг градиентного спуска
            @return metrics: вектор значений метрики на каждом шаге градиентного
            спуска. В случае mae_metric==True вычисляется метрика MAE
            иначе MSE
        """
        n, k = X.shape
        self.init_weights(X.shape[1], y.shape[1])
        metrics = []
        # X_train = np.hstack((X, np.ones((n, 1))))
        for _ in range(num_epochs):
            preds = self.predict(X)
            b_grad = np.mean(2 * (X @ self.W + self.b - y), axis=0)
            W_grad = 2 / X.shape[0] * X.T @ (X @ self.W + self.b - y)
            self.W -= lr * W_grad
            self.b -= lr * np.mean(b_grad, axis=0)
            metrics.append(self.metric(preds, y))
        return metrics

    def predict(self, X):
        """
            Думаю, тут все понятно. Сделайте свои предсказания :)
        """
        # n, k = X.shape
        # X_test = np.hstack((X, np.ones((n, 1))))
        y_pred = X @ self.W + self.b
        return y_pred

    def get_weights(self):
        return self.W


# np.random.seed(42)
# X, Y = datasets.make_regression(n_targets=3, n_features=2, noise=10, random_state=42)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
#
# model = LinearRegression(True)
# mse = model.fit(X_train, Y_train)
# predictions = model.predict(X_test[:, np.newaxis])
# w = model.get_weights()
# plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 0])
#plt.plot(mse)
#plt.show()
# ########################################################
# ########################################################
# def logit(x, w):
#     return np.dot(x, w)
#
# def sigmoid(h):
#     return 1. / (1 + np.exp(-h))


class LogisticRegressionGD:
    '''
    A simple logistic regression for binary classification with gradient descent
    '''

    def __init__(self):
        pass

    def __extend_X(self, X):
        """
            Данный метод должен возвращать следующую матрицу:
            X_ext = [1, X], где 1 - единичный вектор
            это необходимо для того, чтобы было удобнее производить
            вычисления, т.е., вместо того, чтобы считать X@W + b
            можно было считать X_ext@W_ext
        """
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def init_weights(self, input_size, output_size):
        """
            Инициализирует параметры модели
            W - матрица размерности (input_size, output_size)
            инициализируется рандомными числами из
            нормального распределения со средним 0 и стандартным отклонением 0.01
        """
        np.random.seed(42)
        self.W = np.random.normal(0, 0.01, size=(input_size, output_size))

    def get_loss(self, p, y):
        """
            Данный метод вычисляет логистическую функцию потерь
            @param p: Вероятности принадлежности к классу 1
            @param y: Истинные метки
        """
        #p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))

    def get_prob(self, X):
        """
            Данный метод вычисляет P(y=1|X,W)
            Возможно, будет удобнее реализовать дополнительный
            метод для вычисления сигмоиды
        """
        if X.shape[1] != self.W.shape[0]:
            X = self.__extend_X(X)
        return sigmoid(logit(X, self.W))

    def get_acc(self, p, y, threshold=0.5):
        """
            Данный метод вычисляет accuracy:
            acc = \frac{1}{len(y)}\sum_{i=1}^{len(y)}{I[y_i == (p_i >= threshold)]}
        """
        correct = 0
        pred = p >= threshold
        correct = y == pred
        accuracy = (correct.sum()) / len(y)
        return accuracy

    def fit(self, X, y, num_epochs=100, lr=0.001):

        X = self.__extend_X(X)
        self.init_weights(X.shape[1], y.shape[1])

        accs = []
        losses = []
        for _ in range(num_epochs):
            p = self.get_prob(X)

            W_grad = 1 / X.shape[0] * X.T @ (p - y)
            self.W -= lr*W_grad

            # необходимо для стабильности вычислений под логарифмом
            p = np.clip(p, 1e-10, 1 - 1e-10)

            log_loss = self.get_loss(p, y)
            losses.append(log_loss)
            acc = self.get_acc(p, y)
            accs.append(acc)
        return accs, losses


# X, y = datasets.make_blobs(n_samples=10000, n_features=2, centers=2, random_state=42)
# y = y[:, np.newaxis]
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train[:, 0])
# #plt.show()
#
# model = LogisticRegressionGD()
# accs, losses = model.fit(X_train, y_train)
# ########################################################
# ########################################################
def logit(x, w):
    return np.dot(x, w)

def sigmoid(h):
    return 1. / (1 + np.exp(-h))

def batch_generator(X, y, batch_size=100):
    """
        Необходимо написать свой генератор батчей.
        Если вы не знаете, что такое генератор, то, возможно,
        вам поможет
        https://habr.com/ru/post/132554/
        В данном генераторе не надо перемешивать данные
    """
    num_samples = X.shape[0]
    # Заметьте, что в данном случае, если num_samples не делится на batch_size,
    # то последние элементы никогда не попадут в обучение
    # в данном случае нас это не волнует
    num_batches = int(num_samples / batch_size)
    for i in range(num_batches-1):
        # Необходимо отдать batch_size обьектов и соответствующие им target
        yield  X[num_batches*i : num_batches*(i+1)], y[num_batches*i : num_batches*(i+1)]

class LogisticRegressionSGD:
    def __init__(self):
        pass

    def __extend_X(self, X):
        """
            Данный метод должен возвращать следующую матрицу:
            X_ext = [1, X], где 1 - единичный вектор
            это необходимо для того, чтобы было удобнее производить
            вычисления, т.е., вместо того, чтобы считать X@W + b
            можно было считать X_ext@W_ext
        """
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def init_weights(self, input_size, output_size):
        """
            Инициализирует параметры модели
            W - матрица размерности (input_size, output_size)
            инициализируется рандомными числами из
            нормального распределения со средним 0 и стандартным отклонением 0.01
        """
        np.random.seed(42)
        self.W = np.random.normal(0, 0.01, size=(input_size, output_size))

    def get_loss(self, p, y):
        """
            Данный метод вычисляет логистическую функцию потерь
            @param p: Вероятности принадлежности к классу 1
            @param y: Истинные метки
        """
        return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))

    def get_prob(self, X):
        """
            Данный метод вычисляет P(y=1|X,W)
            Возможно, будет удобнее реализовать дополнительный
            метод для вычисления сигмоиды
        """
        if X.shape[1] != self.W.shape[0]:
            X = self.__extend_X(X)
        return sigmoid(logit(X, self.W))

    def get_acc(self, p, y, threshold=0.5):
        """
            Данный метод вычисляет accuracy:
            acc = \frac{1}{len(y)}\sum_{i=1}^{len(y)}{I[y_i == (p_i >= threshold)]}
        """
        correct = 0
        pred = p >= threshold
        correct = y == pred
        accuracy = (correct.sum()) / len(y)
        return accuracy

    def fit(self, X, y, num_epochs=10, lr=0.001):

        X = self.__extend_X(X)
        self.init_weights(X.shape[1], y.shape[1])

        accs = []
        losses = []
        for _ in range(num_epochs):
            gen = batch_generator(X, y)
            for X_, y_ in gen:
                p = self.get_prob(X_)

                W_grad = 1 / X_.shape[0] * X_.T @ (p - y_)
                self.W -= lr*W_grad

                # необходимо для стабильности вычислений под логарифмом
                p = np.clip(p, 1e-10, 1 - 1e-10)

                log_loss = self.get_loss(p, y_)
                losses.append(log_loss)
                acc = self.get_acc(p, y_)
                accs.append(acc)
        return accs, losses



X, y = datasets.make_blobs(n_samples=10000, n_features=2, centers=2, random_state=42)
y = y[:, np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LogisticRegressionSGD()
accs, losses = model.fit(X_train, y_train)
#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train[:, 0])
plt.plot(losses)
plt.show()
