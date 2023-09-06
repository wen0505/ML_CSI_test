import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 載入資料集
df = pd.read_csv('../src_SVM/amp_phase_ave_sd.csv')
df_data = pd.DataFrame(data=df, columns=['amp-ave', 'amp-sd', 'phase-ave', 'phase-sd', 'group'])
# print(df_data)

X = df_data.drop(labels=['group'], axis=1).values        # 移除 group 並取得剩下欄位資料
y = df_data['group']

"""
K-Means
    Parameters:
        n_cluster: K的大小，也就是分群的類別數量。
        random_state: 亂數種子，設定常數能夠保證每次分群結果都一樣。
        n_init: 預設為10次隨機初始化，選擇效果最好的一種來作為模型。
        max_iter: 迭代次數，預設為300代。
    Attributes:
        inertia_: inertia_：float，每個點到其他叢集的質心的距離之和。
        cluster_centers_： 特徵的中心點 [n_clusters, n_features]。
    Methods:
        fit: K個集群分類模型訓練。
        predict: 預測並回傳類別。
        fit_predict: 先呼叫fit()做集群分類，之後在呼叫predict()預測最終類別並回傳輸出。
        transform: 回傳的陣列每一行是每一個樣本到kmeans中各個中心點的L2(歐幾里得)距離。
        fit_transform: 先呼叫fit()再執行transform()。
K-Means 的演算過程:
1. 初始化：指定K個分群，並隨機挑選K個資料點的值當作群組中心值
2. 分配資料點：將每個資料點設為距離最近的中心
3. 計算平均值：重新計算每個分群的中心點
重複步驟2、3，直到資料點不再變換群組為止
"""
kmeansModel = KMeans(n_clusters=2, random_state=46)
clusters_pred = kmeansModel.fit_predict(X)
# -------------------------------------------------------------------
# 評估分群結果
"""
使用者設定 K 個分群後，該演算法快速的找到 K 個中心點並完成分群分類。
擬合好模型後我們可以計算各個 sample 到各該群的中心點的距離之平方和，用來評估集群的成效，其 inertia 越大代表越差。
"""
kmeansModel.inertia_
print(kmeansModel.inertia_)

# 查看各 cluster 的中心，並在圖上畫出
# cluster_centers_： 特徵的中心點 [n_clusters, n_features]。
kmeansModel.cluster_centers_
print(kmeansModel.cluster_centers_)
# -------------------------------------------------------------------
# 真實分類
sns.lmplot(x='amp-ave', y='amp-sd', hue='group', data=df_data, fit_reg=False, legend=False)
plt.legend(title='group', loc='best', labels=['no-people', 'people'])
# -------------------------------------------------------------------
# K-mean 後預測結果
df_data['Predict'] = clusters_pred
sns.lmplot(x='amp-ave', y='amp-sd', data=df_data, hue="Predict", fit_reg=False, legend=False)
plt.scatter(kmeansModel.cluster_centers_[:, 0], kmeansModel.cluster_centers_[:, 1], s=300, c="r", marker='*')
plt.legend(title='group', loc='best', labels=['no-people', 'people'])
# -------------------------------------------------------------------
"""
當你手邊有一群資料，且無法一眼看出有多少個中心的狀況。可用使用下面兩種方法做 k-means 模型評估。
1. Inertia 計算所有點到每群集中心距離的平方和。
2. silhouette scores 側影函數驗證數據集群內一致性的方法。
"""
# 使用 inertia 做模型評估
"""
當K值越來越大，inertia 會隨之越來越小。
正常情況下不會取K最大的，一般是取 elbow point 附近作為 K，即 inertia 迅速下降轉為平緩的那個點。
"""
# k = 1~9 做9次kmeans, 並將每次結果的inertia收集在一個list裡
kmeans_list = [KMeans(n_clusters=k, random_state=46).fit(X) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_list]
print(inertias)
plt.figure(figsize=(8, 3))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow',
             xy=(3, inertias[2]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1))
plt.axis([1, 9, 0, 8000])
# -------------------------------------------------------------------
# 使用 silhouette scores 做模型評估
"""
另外一個方法是用 silhouette scores 去評估，其分數越大代表分群效果越好。
"""
silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_list[1:]]
print(silhouette_scores)
plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "ro-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([2, 9, 0.22, 0.34])
plt.show()
