
from sklearn.ensemble import RandomForestRegressor
import sklearn.pipeline as pl
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
import pandas as pd
import numpy as np
import time
import random
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import json

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tool_logging import init_logger


def create_model():
    model = LogisticRegression(max_iter=1000000)
    return model


def pca_analyze(x_train, x_test):
    # print("pca analyze ", end='')
    x_all = pd.concat([x_train, x_test])
    pca = PCA(n_components=60).fit_transform(x_all)
    x_train_pca = pca[:len(x_train)]
    x_test_pca = pca[len(x_train):]
    # print("over")
    return x_train_pca, x_test_pca


def train_and_predict(folds=1, data_indices=None):
    if data_indices is None:
        data_indices = list(range(0, folds))
    scores = {}
    for data_index in data_indices:
        print(f"read the index {data_index} ", end='')
        train_data = pd.read_excel(f"data/train_data/train_data{data_index}.xlsx")
        test_data = pd.read_excel(f"data/train_data/test_data{data_index}.xlsx")
        print("over ！")
        scores[data_index] = {}

        ADMET_columns = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
        for ADMET_column in ADMET_columns:
            # ADMET_column = ADMET_columns[4]
            x_test = test_data.iloc[:, :-6]
            y_test = test_data.loc[:, ADMET_column]
            x_train = train_data.iloc[:, :-6]
            y_train = train_data.loc[:, ADMET_column]
            x_train_pca, x_test_pca = pca_analyze(x_train, x_test)


            print(f"{data_index}#{ADMET_column}:", end='')
            # 模型===========
            model = create_model()
            model.fit(x_train_pca, y_train)

            # 预测
            predict = model.predict(x_test_pca)
            score = accuracy_score(predict, y_test)
            scores[data_index][ADMET_column] = score
            print(f"\t{score}")

    for data_index in data_indices:
        print(data_index, ': ', scores[data_index])
    return scores



if __name__ == "__main__":
    logging = init_logger(log_dir='./log', log_file='train_and_predict_question3.txt', mode='a')

    scores = train_and_predict(folds=2, data_indices=None)  # 重点

    logging.info(scores)



