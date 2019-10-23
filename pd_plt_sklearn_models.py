import sys
import os
import numpy as np
import pandas as pd
import pyqtgraph as pg
import statsmodels.api as sm
import patsy as pt
import sklearn.linear_model as lm
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt


def print_metrics(y_preds, y):
    print(f'R^2: {r2_score(y_preds, y)}')
    print(f'MSE: {mean_squared_error(y_preds, y)}')

try:
    # df = pd.read_csv("ExpertDoctors.csv", delimiter=',')
    #
    # x = df.iloc[:, :-1]
    # y = df.iloc[:, -1]
    #
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8)
    #
    # lr = LinearRegression()
    # lr.fit(X_train, y_train)
    # print_metrics(lr.predict(X_test), y_test)
    # print("#"*10)
    # rlr = Ridge(alpha=0.5)
    # rlr.fit(X_train, y_train)
    # print_metrics(rlr.predict(X_test), y_test)
    # print("#" * 10)
    # gbr = GradientBoostingRegressor()
    # gbr.fit(X_train, y_train)
    # print_metrics(gbr.predict(X_test), y_test)
    # print("#" * 10)
    # rfr = RandomForestRegressor()
    # rfr.fit(X_train, y_train)
    # print_metrics(rfr.predict(X_test), y_test)
    ###########################################################################
    #df = pd.read_csv("BlackFriday.csv", delimiter=',')
    #print(df)
    #print(df.isna().mean())
    #print(df['Product_Category_3'].isna().sum())
    # print(df['User_ID'].count())
    # con = len(df[df['Gender'] == 'M'][df['Age'] == '26-35'])
    #
    # con2 = len(df[df['Gender'] == 'F']) - \
    #        len(df[df['Gender'] == 'F'][(df['Age'] == '0-17')
    #                                   |(df['Age'] == '18-25')
    #                                   |(df['Age'] == '26-35')])
    # print((con + con2)/df['User_ID'].count())

    #ages = df[['Age', 'Gender', 'Purchase']][df['Age'] == "46-50"][df['Gender'] == "F"]
    #print(len(ages[ages['Purchase'] > 20000]))
    # print(len(ages))
    #citA = df[['City_Category', 'Gender']][df['City_Category'] == "A"]
    #print(citA[:][citA['City_Category'] == "A"])
    #print(len(citA[:][citA['Gender'] == "M"]))
    ############################################################################

    all_data = pd.read_csv('forest_dataset.csv', )
    print(all_data.head())
    labels = all_data[all_data.columns[-1]].values
    feature_matrix = all_data[all_data.columns[:-1]].values
    train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=42)

    clf = KNeighborsClassifier()
    clf.fit(train_feature_matrix, train_labels)
    pred_labels = clf.predict(test_feature_matrix)
    print(accuracy_score(test_labels, pred_labels))

    params = {'weights': ['uniform', 'distance'], 'n_neighbors': np.arange(1, 10), 'metric':['manhattan', 'euclidean']}

    clf_grid = GridSearchCV(clf, params, cv=5, scoring='accuracy', n_jobs=-1)
    clf_grid.fit(train_feature_matrix, train_labels)
    print(clf_grid.best_params_)

    optimal_clf = KNeighborsClassifier(n_neighbors=4)
    optimal_clf.fit(train_feature_matrix, train_labels)
    pred_prob = optimal_clf.predict_proba(test_feature_matrix)

    unique, freq = np.unique(test_labels, return_counts=True)
    freq = list(map(lambda x: x / len(test_labels), freq))

    pred_freq = pred_prob.mean(axis=0)
    print(pred_freq)
    plt.figure(figsize=(10, 8))
    plt.bar(range(1, 8), pred_freq, width=0.4, align="edge", label='prediction')
    plt.bar(range(1, 8), freq, width=-0.4, align="edge", label='real')
    plt.legend()
    plt.show()

    # # создание модели с указанием гиперпараметра C
    # clf = LogisticRegression(C=1)
    # # обучение модели
    # clf.fit(train_feature_matrix, train_labels)
    # # предсказание на тестовой выборке
    # y_pred = clf.predict(test_feature_matrix)
    # print(accuracy_score(test_labels, y_pred))
    #
    # # заново создадим модель, указав солвер
    # clf = LogisticRegression(solver='saga')
    # # опишем сетку, по которой будем искать
    # param_grid = {
    #     'C': np.arange(1, 5),  # также можно указать обычный массив, [1, 2, 3, 4]
    #     'penalty': ['l1', 'l2'],
    # }
    # # создадим объект GridSearchCV
    # search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=5, refit=True, scoring='accuracy')
    # # запустим поиск
    # search.fit(feature_matrix, labels)
    # # выведем наилучшие параметры
    # print(search.best_params_)

except Exception:
    print("No data!")
