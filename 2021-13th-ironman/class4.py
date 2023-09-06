import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# 讀取 csv 檔
df = pd.read_csv('C:/Users/lab517/PycharmProjects/svm_csi/src_SVM/amp_phase_ave_sd.csv')
df_data = pd.DataFrame(data=df, columns=['amp-ave', 'amp-sd', 'phase-ave', 'phase-sd', 'group'])
# print(df_data)
# -------------------------------------------------------------------
# 檢查缺失值
"""
使用 numpy 所提供的函式來檢查是否有 NA 缺失值，假設有缺失值使用 drop() 來移除。
使用的時機在於當只有少量的缺失值適用，若遇到有大量缺失值的情況，或是本身的資料量就很少的情況下建議可以透過機器學習的方法補值來預測缺失值。
"""
X = df_data.drop(labels=['group'], axis=1).values     # 移除 group 並取得剩下欄位資料
y = df_data['group']
# checked missing data
# print("checked missing data(NAN mount):", len(np.where(np.isnan(X))[0]))
# -------------------------------------------------------------------
# 切割訓練集與測試集
"""
我們透過 Sklearn 所提供的 train_test_split() 方法來為我們的資料進行訓練集與測試集的切割。在此方法中我們可以設定一些參數來讓我們切割的資料更多樣性。
其中 test_size 參數就是設定測試集的比例，範例中我們設定 0.3 即代表訓練集與測試集的比例為 7:3。
另外預設資料切割的方式是隨機切割 shuffle=True 對原始數據進行隨機抽樣，以保證隨機性。
若想要每次程式執行時切割結果都是一樣的可以設定亂數隨機種子 random_state 並給予一個隨機數值。
最後一個是 stratify 分層隨機抽樣，特別是在原始數據中樣本標籤分佈不均衡時非常有用。
使用時機是確保分類問題 y 的類別數量分佈要與原資料集一致。以免資料集切割不平均導致模型訓練時有很大的偏差。
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

print('train shape:', X_train.shape)
print('test shape:', X_test.shape)
# -------------------------------------------------------------------
# Standardization 平均&變異數標準化
"""
將所有特徵標準化，也就是高斯分佈。使得數據的平均值為 0，方差為 1。
適合的使用時機於當有些特徵的方差過大時，使用標準化能夠有效地讓模型快速收斂。
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# scaled之後的資料零均值，單位方差
print('資料集 X 的平均值 : ', X_train.mean(axis=0))
print('資料集 X 的標準差 : ', X_train.std(axis=0))

print('\nStandardScaler 縮放過後訓練集的平均值 : ', X_train_scaled.mean(axis=0))
print('StandardScaler 縮放過後訓練集的標準差 : ', X_train_scaled.std(axis=0))

# 訓練集的 Scaler 擬合完成後，我們就能做相同的轉換在測試集上。
X_test_scaled = scaler.transform(X_test)

print('\nStandardScaler 縮放過後測試集的平均值 : ', X_test_scaled.mean(axis=0))
print('StandardScaler 縮放過後測試集的標準差 : ', X_test_scaled.std(axis=0))

# 如果想將轉換後的資料還原可以使用 inverse_transform() 將數值還原成原本的輸入。
X_test_inverse = scaler.inverse_transform(X_test_scaled)
# -------------------------------------------------------------------
# MinMaxScaler最小最大值標準化
"""
在 MinMaxScaler 中是給定了一個明確的最大值與最小值。每個特徵中的最小值變成了0，最大值變成了1。數據會縮放到到[0,1]之間。
"""
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# scaled 之後的資料最小值、最大值
print('資料集 X 的最小值 : ', X_train.min(axis=0))
print('資料集 X 的最大值 : ', X_train.max(axis=0))

print('\nStandardScaler 縮放過後訓練集的最小值 : ', X_train_scaled.min(axis=0))
print('StandardScaler 縮放過後訓練集的最大值 : ', X_train_scaled.max(axis=0))

X_test_scaled = scaler.transform(X_test)

print('\nStandardScaler 縮放過後測試集的最小值 : ', X_test_scaled.min(axis=0))
print('StandardScaler 縮放過後測試集的最大值 : ', X_test_scaled.max(axis=0))
# -------------------------------------------------------------------
# RobustScaler
"""
可以有效的縮放帶有 outlier 的數據，透過 Robust 如果數據中含有異常值在縮放中會捨去。
"""
scaler = RobustScaler().fit(X)
X_scaled = scaler.transform(X)

X_test_scaled = scaler.transform(X_test)
# print('\nRobustScaler : \n', X_test_scaled)
