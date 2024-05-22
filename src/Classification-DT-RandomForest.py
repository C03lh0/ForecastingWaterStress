import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import gzip

import seaborn as sns
from typing import Tuple
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


def cross_val_predict(model, kfold: KFold, X: np.array, y: np.array) -> Tuple[np.array, np.array, np.array]:
    model_ = cp.deepcopy(model)

    no_classes = len(np.unique(y))

    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes])

    for train_ndx, test_ndx in kfold.split(X):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba


def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):
    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)

    plt.figure(figsize=(12.8, 6))
    sns.set(font_scale=1.8)
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, annot_kws={"size": 20}, cmap="Blues", fmt="g")
    plt.xlabel('Classified as', fontsize = 25, weight='bold');
    plt.ylabel('Actual', fontsize = 25, weight='bold');
    #plt.title('Confusion Matrix', fontsize = 30)

    plt.show()


f = gzip.open("data/mydata.csv.gz")
df_reviews = pd.read_csv(f, low_memory=False)

df_reviews.dropna(inplace=True)
df_reviews.reset_index(drop=True, inplace=True)

dataFrameX = df_reviews.drop(columns='label')
features = list(dataFrameX.columns.values)

X = df_reviews.loc[:, ['Rds', 'DIgs', 'tds', 'tgs']]
y = df_reviews['label']

model= DecisionTreeClassifier(criterion='gini')
#model = RandomForestClassifier(n_estimators=100, random_state=42)
kfold = KFold(n_splits=5, random_state=42, shuffle=True)

cv_results = cross_val_score(model, X, y, cv=kfold, scoring='f1_weighted', verbose=10)

print(cv_results.mean(), cv_results.std())

actual_classes, predicted_classes, _ = cross_val_predict(model, kfold, X.to_numpy(), y.to_numpy())
plot_confusion_matrix(actual_classes, predicted_classes, ["healthy", "uncertain", "stress", "recovery"])