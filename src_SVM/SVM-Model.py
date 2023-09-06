"""
根據 data.csv 文件中給定的數據創建支持向量機 (SVM) 的程式碼。
可以找到具有最大邊距的兩組數據之間的區別。
可以找到一個線性超平面來表徵不同的組並將其顯示到圖片上。
參考 : https://github.com/nihaal-prasad/SVM-Model
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, metrics

FILE_NAME1 = "C:/Users/lab517/PycharmProjects/svm_csi/src_SVM/amp_sd.csv"

# -----------------------------------------------------------------------------
# (1) 載入資料集
data_csv = pd.read_csv(FILE_NAME1)

# Our first step is to read the given data
x_title = list(data_csv)[0]
y_title = list(data_csv)[1]
group_title = list(data_csv)[2]

# Specify inputs for the model
xy = data_csv[[x_title, y_title]].values
type_label = np.where(data_csv[group_title] == 1, 1, 2)

# Create and fit the linear SVN model
model = svm.SVC(kernel='linear')
model.fit(xy, type_label)

# 使用訓練資料預測分類
predicted = model.predict(xy)
# 計算訓練集 MSE 誤差
mse = metrics.mean_squared_error(type_label, predicted)
print('訓練集 MSE: ', mse)

# Get the hyperplane
# w = model.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(min(xy[:, 0]), max(xy[:, 1]))
# yy = a * xx - (model.intercept_[0] / w[1])

# Visualize the results
sns.lmplot(x=x_title, y=y_title, data=data_csv, hue=group_title, fit_reg=False, legend=False)     # Plot the points
# plt.plot(xy, type_label, linewidth=2, color='black')        # Plot the hyperplane
plt.legend(title='target', loc='best', labels=['no-people', 'people'])
plt.title('indoor state')
plt.show()
