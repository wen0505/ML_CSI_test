import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 載入資料集
df = pd.read_csv('../src_SVM/amp_phase_ave_sd.csv')
df_data = pd.DataFrame(data=df, columns=['amp-ave', 'amp-sd', 'phase-ave', 'phase-sd', 'group'])
# print(df_data)

# 切割訓練集與測試集
X = df_data.drop(labels=['group'], axis=1).values           # 移除 group 並取得剩下欄位資料
y = df_data['group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print('train shape:', X_train.shape)
print('test shape:', X_test.shape)
# -------------------------------------------------------------------
# PCA
"""
PCA
    Parameters:
        n_components: 指定PCA降維後的特徵維度數目。
        whiten: 是否進行白化True/False。白化意指，對降維後的數據的每個特徵進行正規化，即讓方差都為1、平均值為0。默認值為False。
        random_state: 亂數種子，設定常數能夠保證每次PCA結果都一樣。
    Attributes:
        explained_variance_： array類型。降維後的各主成分的方差值，主成分方差值越大，則說明這個主成分越重要
        explained_variance_ratio_： array類型。降維後的各主成分的方差值佔總方差值的比例，主成分所佔比例越大，則說明這個主成分越重要。
        n_components_： int類型。返回保留的特徵個數。
    Methods:
        fit(X,y)：把數據放入模型中訓練模型。
        fit_transform(X,[,y])all：訓練模型同時返回降維後的數據。
        transform(X)：對於訓練好的數據降維。
"""
pca = PCA(n_components=2, iterated_power=1)
train_reduced = pca.fit_transform(X_train)

print('PCA方差比: ', pca.explained_variance_ratio_)
print('PCA方差值:', pca.explained_variance_)

plt.figure(figsize=(8, 6))
plt.scatter(train_reduced[:, 0], train_reduced[:, 1], c=y_train, alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()

test_reduced = pca.transform(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(test_reduced[:, 0], test_reduced[:, 1], c=y_test, alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()
# -------------------------------------------------------------------
# t-SNE
"""
t-SNE使用了更複雜的公式來表達高維與低維之間的關係。且能夠允許非線性的轉換。
    Parameters:
        n_components: 指定t-SNE降維後的特徵維度數目。
        n_iter: 設定迭代次數。
        random_state: 亂數種子，設定常數能夠保證每次t-SNE結果都一樣。
    Attributes:
        explained_variance_： array類型。降維後的各主成分的方差值，主成分方差值越大，則說明這個主成分越重要
        explained_variance_ratio_： array類型。降維後的各主成分的方差值佔總方差值的比例，主成分所佔比例越大，則說明這個主成分越重要。
        n_components_： int類型。返回保留的特徵個數。
    Methods:
        fit(X,y)：把數據放入模型中訓練模型。
        fit_transform(X)：訓練模型同時返回降維後的數據。
        transform(X)：對於訓練好的數據降維。
"""
tsneModel = TSNE(n_components=2, random_state=42, n_iter=1000)
train_reduced = tsneModel.fit_transform(X_train)
plt.figure(figsize=(8, 6))
plt.scatter(train_reduced[:, 0], train_reduced[:, 1], c=y_train, alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()
plt.show()
