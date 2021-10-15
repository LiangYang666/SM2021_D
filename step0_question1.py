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

#%%
variable_file = "data/origin_data/Molecular_Descriptor.xlsx"
pIC50_file = "data/origin_data/ERα_activity.xlsx"
ADMET_file = "data/origin_data/ADMET.xlsx"
#%%

if __name__ == "__main__":
    variables = pd.read_excel(variable_file)
    pIC50s = pd.read_excel(pIC50_file)
    ADMETs = pd.read_excel(ADMET_file)
    print("读取完成")
    #%%
    variables.describe()
    columns_0 = []
    variables_del = pd.DataFrame()
    for column in variables.columns:
        if (variables[column]==0).all():
            columns_0.append(column)
            # print(column)
        else:
            variables_del[column] = variables[column]
    variables_del.to_excel(f"data/train_data/Molecular_Descriptor_删除全0的{columns_0.__len__()}列.xlsx", index=None)
    pd.DataFrame(columns_0).to_excel(f"data/train_data/删除全0的列名.xlsx", index=None)
    columns_0.__len__()
    #%%
    pIC50s.describe()
    #%%
    train_data = variables_del.iloc[:, 1:]
    ADMET_columns = ['Caco-2', 'CYP3A4', 'hERG', 'HOB', 'MN']
    train_data['pIC50'] = pIC50s.iloc[:, 2]
    SMILES = pIC50s.iloc[:, 0]
    for ADMET_column in ADMET_columns:
        train_data[ADMET_column] = ADMETs.loc[:, ADMET_column]
    x_train_origin, y_train_origin = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    #%%
    train_data.to_excel("data/train_data/train_data删除全0.xlsx", index=None)
    print('存储完成')
    #%%
    # 归一化的
    train_data_scale = train_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    x_train_scale = train_data_scale.iloc[:, :-6]
    y_train_scale = train_data_scale.loc[:, 'pIC50']
    print("随机森林拟合")
    forest_scale = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)
    print("begin fit")
    time_start = time.time()
    forest_scale.fit(x_train_scale, y_train_scale)
    time_end = time.time()
    print(f"fit over {time_end-time_start}")

    #%%
    # 归一化的值 随机森林取值
    importances = forest_scale.feature_importances_
    indices = np.argsort(importances)[::-1]
    importances_pd = pd.DataFrame(columns=['importances', 'column index', 'column'])
    columns_del = x_train_scale.columns
    for f in range(x_train_scale.shape[1]):
        # print(f"{f+1})\t{indices[f]}\t{importances[indices[f]]:1.5f}\t{SMILES[indices[f]]} ")
        importances_pd.loc[f] = [importances[indices[f]], indices[f],columns_del[indices[f]]]
    importances_pd.head(20)
    print('存储所有变量 按重要性排序')
    importances_pd.to_excel("data/question1/importances_pd.xlsx", index=None)

    #%%
    # 归一化后的 灰色关联分析
    def GRA_ONE(DataFrame,m):
        gray= DataFrame
        #读取为df格式
        gray=(gray - gray.min()) / (gray.max() - gray.min())
        #标准化
        std=gray.iloc[:,m]#为标准要素
        ce=gray.iloc[:, 0:]#为比较要素
        n=ce.shape[0]
        m=ce.shape[1]#计算行列

        #与标准要素比较，相减
        a=np.zeros([m,n])
        for i in range(m):
            for j in range(n):
                a[i,j]=abs(ce.iloc[j,i]-std[j])

        #取出矩阵中最大值与最小值
        c=np.amax(a)
        d=np.amin(a)

        #计算值
        result=np.zeros([m,n])
        for i in range(m):
            for j in range(n):
                result[i,j]=(d+0.5*c)/(a[i,j]+0.5*c)

        #求均值，得到灰色关联值
        result2=np.zeros(m)
        for i in range(m):
                result2[i]=np.mean(result[i,:])
        RT=pd.DataFrame(result2)
        return RT

    def GRA(DataFrame):
        list_columns = [str(s) for s in range(len(DataFrame.columns)) if s not in [None]]
        df_local = pd.DataFrame(columns=list_columns)
        for i in range(len(DataFrame.columns)):
            if i!=len(DataFrame.columns)-1:
                continue
            df_local.iloc[:,i] = GRA_ONE(DataFrame,m=i)[0]
        return df_local

    # data_gra = GRA(train_data_scale)
    data_gra = GRA_ONE(train_data_scale, train_data_scale.columns.__len__()-1)

    #%%
    data_gra['column'] = train_data_scale.columns
    data_gra = data_gra.sort_values(by=[0], ascending=False)
    data_gra.to_excel("data/question1/gray.xlsx")
    data_gra.head(20)
