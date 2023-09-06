import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets, preprocessing, svm, metrics

FILE_NAME1 = "C:/Users/lab517/PycharmProjects/svm_csi/src_SVM/amp_sd.csv"

# -----------------------------------------------------------------------------
# (1) 載入資料集
data_csv = pd.read_csv(FILE_NAME1)
# df_data = pd.DataFrame(data=data_csv['data'], columns=['ave', 'sd'])
# print(df_data)

# X = df_data['no-people'].values
# y = df_data['people'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# print('train shape:', X_train.shape)
# print('test shape:', X_test.shape)
# print('train shape:', y_train.shape)
# print('test shape:', y_test.shape)
# -----------------------------------------------------------------------------
# Preprocessing_data = preprocessing.LabelEncoder()
# Group_ = Preprocessing_data.fit_transform(list(data_csv["group"]))
# ave_1 = Preprocessing_data.fit_transform(list(data_csv["ave"]))
# sd_1 = Preprocessing_data.fit_transform(list(data_csv["sd"]))
# predict = "group_"

# X = list(zip(ave_1, sd_1))
# Y = list(Group_)

# x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size=0.3)
# print('train shape:', x_train.shape)
# print('test shape:', x_test.shape)
# print('train shape:', y_train.shape)
# print('test shape:', y_test.shape)
