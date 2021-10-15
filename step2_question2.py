
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
from tool_ogging import init_logger


# 设置模型
def create_model():
    model = pl.make_pipeline(
        sp.PolynomialFeatures(8),  # 多项式特征拓展器
        lm.LinearRegression()  # 线性回归器
    )
    # model = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)
    # model = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)
    return model


# 选出需要的列
def get_important_columns():
    # importances_pd = pd.read_excel("data/question1/gray.xlsx")
    importances_pd = pd.read_excel("data/question1/importances_pd.xlsx")
    importances_20 = importances_pd.head(20)
    columns = importances_20['column'].to_list()
    return columns


def train_and_predict_question2(folds=1, data_indices=None):
    if data_indices is None:
        data_indices = list(range(0, folds))
    errors = {}

    for data_index in data_indices:
        print(f"read the index {data_index} ", end='')
        train_data = pd.read_excel(f"data/train_data/train_data{data_index}.xlsx")
        test_data = pd.read_excel(f"data/train_data/test_data{data_index}.xlsx")
        print("over ！")

        # select_columns
        # %%
        # train_data_scale = train_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        train_data = pd.read_excel("data/train_data/train_data0.xlsx")
        test_data = pd.read_excel("data/train_data/test_data0.xlsx")
        # %%
        select_columns = get_important_columns()

        x_train0 = train_data[select_columns]
        y_train0 = train_data.loc[:, 'pIC50']

        x_test0 = test_data[select_columns]
        y_test0 = test_data.loc[:, 'pIC50']

        model = create_model()
        # %%
        print(f"{data_index} ", end='')
        print("begin fit", end='')
        time_start = time.time()
        model.fit(x_train0, y_train0)
        time_end = time.time()
        print(f"over {time_end - time_start:.3f}", end='')
        # %%
        pred_test = model.predict(x_test0)
        error = mean_squared_error(y_test0, pred_test)
        print(error)
        errors[data_index] = error
        print(f"\terror: {error}")
    print(errors)
    return errors


if __name__ == "__main__":
    logging = init_logger(log_dir='data/question2', log_file='train_and_predict_question2.txt', mode='a')

    errors = train_and_predict_question2(folds=2, data_indices=None)     # 重点

    logging.info(errors)

