"""
no use
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

# 載入資料集
df = pd.read_csv('C:/Users/lab517/PycharmProjects/svm_csi/src_SVM/amp_phase_ave_sd.csv')
# df_data = pd.DataFrame(data=df, columns=['amp-ave', 'amp-sd', 'phase-ave', 'phase-sd', 'group'])
# print(df_data)

X = df.data
b = np.ones((X.shape[0], 1))
X = np.hstack((X, b))
y = df.target

Beta = np.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ Beta

print('MSE:', mean_squared_error(y_pred, y))
# print(X)
# print(y)

# -------------------------------------------------------------------
