import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

df = pd.read_csv('C:/Users/lab517/PycharmProjects/svm_csi/src_SVM/amp_phase_ave_sd.csv')
df_data = pd.DataFrame(data=df, columns=['amp-ave', 'amp-sd', 'phase-ave', 'phase-sd', 'group'])
# print(df_data)
# -------------------------------------------------------------------
# 切割訓練集與測試集
X = df_data.drop(labels=['group'], axis=1).values       # 移除 group 並取得剩下欄位資料
y = df_data['group'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print('train shape:', X_train.shape)
print('test shape:', X_test.shape)
# -------------------------------------------------------------------
"""
K-近鄰演算法 (KNN)
KNN 的全名 K Nearest Neighbor 是屬於機器學習中的 Supervised learning 其中一種算法，顧名思義就是 k 個最接近你的鄰居。
分類的標準是由鄰居「多數表決」決定的。在 Sklearn 中 KNN 可以用作分類或迴歸的模型。

KNN 分類器
在分類問題中 KNN 演算法採多數決標準，利用 k 個最近的鄰居來判定新的資料是在哪一群。其演算法流程非常簡單，
首先使用者先決定 k 的大小。接著計算目前該筆新的資料與鄰近的資料間的距離。
第三步找出跟自己最近的 k 個鄰居，查看哪一組鄰居數量最多，就加入哪一組。
1. 決定 k 值
2. 求每個鄰居跟自己之間的距離
3. 找出跟自己最近的 k 個鄰居，查看哪一組鄰居數量最多，就加入哪一組
如果還是沒辦法決定在哪一組，回到第一步調整 k 值，再繼續
k 的大小會影響模型最終的分類結果。

建立 k-nearest neighbors(KNN) 模型
    Parameters:
        n_neighbors: 設定鄰居的數量(k)，選取最近的k個點，預設為5。
        algorithm: 搜尋數演算法{'auto'，'ball_tree'，'kd_tree'，'brute'}，可選。
        metric: 計算距離的方式，預設為歐幾里得距離。
    Attributes:
        classes_: 取得類別陣列。
        effective_metric_: 取得計算距離的公式。
    Methods:
        fit: 放入X、y進行模型擬合。
        predict: 預測並回傳預測類別。
        score: 預測成功的比例。
"""
# 建立 KNN 模型
knnModel = KNeighborsClassifier(n_neighbors=3)
# 使用訓練資料訓練模型
knnModel.fit(X_train, y_train)
# 使用訓練資料預測分類
predicted = knnModel.predict(X_train)

# 預測成功的比例
print('訓練集: ', knnModel.score(X_train, y_train))
print('測試集: ', knnModel.score(X_test, y_test))
# -------------------------------------------------------------------
# 測試集真實分類
# 建立測試集的 DataFrame
df_test = pd.DataFrame(X_test, columns=['amp-ave', 'amp-sd', 'phase-ave', 'phase-sd'])
df_test['group'] = y_test
pred = knnModel.predict(X_test)
df_test['Predict'] = pred

sns.lmplot(x="amp-ave", y="amp-sd", hue='group', data=df_test, fit_reg=False, legend=False)
plt.legend(title='target', loc='upper left', labels=['unmanned', 'manned'])
plt.show()
# -------------------------------------------------------------------
# KNN (測試集)預測結果
sns.lmplot(x="amp-ave", y="amp-sd", hue='Predict', data=df_test, fit_reg=False, legend=False)
plt.legend(title='target', loc='upper left', labels=['unmanned', 'manned'])
plt.show()
# -------------------------------------------------------------------
"""
no use
進階學習
查看不同的K分類結果
為了方便視覺化我們將原有的測試集特徵使用PCA降成2維。接著觀察在不同 K 的狀況下，分類的情形為何。
"""
def plot_decision_regions(X, y, classifier, test_idx = None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=[cmap(idx)], marker=markers[idx], label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

def knn_model(plot_dict, X, y, k):
    #create model
    model = KNeighborsClassifier(n_neighbors=k)

    #training
    model.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    if k in plot_dict:
        plt.subplot(plot_dict[k])
        plt.tight_layout()
        plot_decision_regions(X, y, model)
        plt.title('Plot for K: %d'%k)
from sklearn.decomposition import PCA
pca = PCA(n_components=2, iterated_power=1)
train_reduced = pca.fit_transform(X_train)
test_reduced = pca.fit_transform(X_test)
# -------------------------------------------------------------------
# KNN 訓練集 PCA 2 features
plt.figure(figsize=(8.5, 6))

# 調整 K
plot_dict = {1: 231, 2: 232, 3: 233, 6: 234, 10: 235, 15: 236}
for i in plot_dict:
    knn_model(plot_dict, train_reduced, y_train, i)
