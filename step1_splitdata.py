
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

if __name__ == "__main__":
    train_data = pd.read_excel("data/train_data/train_data删除全0.xlsx")
    # 划分数据集
    indices_list = list(range(len(train_data)))
    indices_list_random = indices_list.copy()
    random.shuffle(indices_list_random)
    ratio = 0.7
    for i in tqdm(range(10)):
        random.seed(i)
        train_indices = indices_list_random[:int(ratio*len(indices_list_random))]
        test_indices = indices_list_random[int(ratio*len(indices_list_random)):]
        train_data_divide = train_data.iloc[train_indices]
        test_data_divide = train_data.iloc[test_indices]
        train_data_divide.to_excel(f"data/train_data/train_data{i}.xlsx", index=None)
        test_data_divide.to_excel(f"data/train_data/test_data{i}.xlsx", index=None)
