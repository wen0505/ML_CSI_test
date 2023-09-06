# https://github.com/andy6804tw/2021-13th-ironman/tree/main/3.%E4%BD%A0%E7%9C%9F%E4%BA%86%E8%A7%A3%E8%B3%87%E6%96%99%E5%97%8E%EF%BC%9F%E8%A9%A6%E8%A9%A6%E7%9C%8B%E8%A6%96%E8%A6%BA%E5%8C%96%E5%88%86%E6%9E%90%E5%90%A7%EF%BC%81
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# 讀取 csv 檔
# df = pd.read_csv('../src_SVM/test1.csv')
# df_data = pd.DataFrame(data=df, columns=['ave', 'sd', 'group'])

df = pd.read_csv('C:/Users/lab517/PycharmProjects/svm_csi/src_SVM/amp_phase_ave_sd.csv')
df_data = pd.DataFrame(data=df, columns=['amp-ave', 'amp-sd', 'phase-ave', 'phase-sd', 'group'])
# print(df_data)
# -------------------------------------------------------------------
# 直方圖 histograms
"""
直方圖是一種對數據分布情況的圖形表示，是一種二維統計圖表。我們可以直接呼叫 Pandas 內建函式 hist() 進行直方圖分析。
其中我們可以設定 bins(箱數)，預設值為 10。如果設定的輸量越大，其代表需要分割的精度越細。通常取一個適當的箱數即可觀察該特徵在資料集中的分佈情況。
藉由直方圖我們可以知道每個值域的分佈大小與數量。也能發現輸出項的類別共有三個，並且這三個類別的數量都剛好各有 50 筆資料。
我們也能得知這一份資料集的輸出類別是一個非常均勻的資料。
"""
df_data.hist(alpha=0.6, layout=(3, 2), figsize=(12, 8), bins=10)
plt.tight_layout()

# 也可以透過 Seaborn 的 histplot 做出更詳細的直方圖分析。並利用和密度估計 kde=True 來查看每個特徵的分佈狀況
fig, axes = plt.subplots(nrows=1, ncols=4)
fig.set_size_inches(15, 4)
sns.histplot(df_data["amp-ave"][:], ax=axes[0], kde=True)
sns.histplot(df_data["amp-sd"][:], ax=axes[1], kde=True)
sns.histplot(df_data["phase-ave"][:], ax=axes[2], kde=True)
sns.histplot(df_data["phase-sd"][:], ax=axes[3], kde=True)
# -------------------------------------------------------------------
# 核密度估計
"""
核密度估計分爲兩部分，分別有對角線部分和非對角線部分。
在對角線部分是以核密度估計圖（Kernel Density Estimation）的方式呈現，也就是用來看某一個特徵的分佈情況，x軸對應著該特徵的數值，y軸對應著該特徵的密度也就是特徵出現的頻率。
在非對角線的部分為兩個特徵之間分佈的關聯散點圖。將任意兩個特徵進行配對，以其中一個爲橫座標，另一個爲縱座標，將所有的數據點繪製在圖上，用來衡量兩個變量的關聯程度。
diag_kind：控制對角線上的圖的類型，可選擇"hist"或"kde"
"""
# 使用 Pandas 繪製
scatter_matrix(df_data, figsize=(10, 10), color='b', diagonal='kde')

# 使用 Seaborn  繪製
# sns.pairplot(df_data, hue="group", palette="pastel", height=2, diag_kind="kde")
# -------------------------------------------------------------------
# 關聯分析
"""
透過 pandas 的 corr() 函式可以快速的計算每個特徵間的彼此關聯程度。
其區間值為-1~1之間，數字越大代表關聯程度正相關越高。
相反的當負的程度很高我們可以解釋這兩個特徵之間是有很高的負關聯性。
"""
# correlation 計算
corr = df_data[['amp-ave', 'amp-sd', 'phase-ave', 'phase-sd', 'group']].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr, square=True, annot=True, cmap="RdBu_r")
# -------------------------------------------------------------------
# 散佈圖
"""
透過散佈圖我們可以從二維的平面上觀察兩兩特徵間彼此的分佈狀況。如果該特徵重要程度越高，群聚的效果會更加顯著。
"""
sns.lmplot(x='amp-ave', y='amp-sd', hue='group', data=df_data, fit_reg=False, legend=False)
plt.legend(title='group', loc='best', labels=['no-people', 'people'])

sns.lmplot(x='phase-ave', y='phase-sd', hue='group', data=df_data, fit_reg=False, legend=False)
plt.legend(title='group', loc='best', labels=['no-people', 'people'])
# -------------------------------------------------------------------
# 箱形圖
"""
透過箱形圖可以分析每個特徵的分布狀況以及是否有離群值。
"""
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 6), sharey=False)
axes[0].set_title('amp-ave')
sns.boxplot(df_data['amp-ave'], ax=axes[0], color='pink', width=0.5, showmeans=True)

axes[1].set_title('amp-sd')
sns.boxplot(df_data['amp-sd'], ax=axes[1], color='lightblue', width=0.5, showmeans=True)

axes[2].set_title('phase-ave')
sns.boxplot(df_data['phase-ave'], ax=axes[2], color='lightgreen', width=0.5, showmeans=True)

axes[3].set_title('phase-sd')
sns.boxplot(df_data['phase-sd'], ax=axes[3], color='mediumslateblue', width=0.5, showmeans=True)

axes[4].set_title('group')
sns.boxplot(df_data['group'], ax=axes[4], color='lightyellow', width=0.5, showmeans=True)

plt.show()
