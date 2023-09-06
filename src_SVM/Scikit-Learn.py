"""
通過四個屬性來判斷室內是不是有人，四個屬性為振福的平均值、振幅的標準差、相位的平均值和相位的標準差。
我們將使用 SVM 解決這個二分類問題。剩下部分是標準的機器學習流程。
參考 : https://itw01.com/QU5YUEM.html
參考 : https://github.com/smurlafries1/MachineLearningPhishingDetection
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

FILE_NAME1 = "C:/Users/lab517/PycharmProjects/svm_csi/src_SVM/amp_phase_ave_sd.csv"

# (1) 載入資料集
data_csv = pd.read_csv(FILE_NAME1)

print("Reading the dataset...")

# 把屬性和類別標籤分開
X = data_csv.drop('group', axis=1)
y = data_csv['group']

# 把資料分成訓練和測試兩部分
print("Dividing the dataset into testing and training sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print('train shape:', X_train.shape)
print('test shape:', X_test.shape)
print('train shape:', y_train.shape)
print('test shape:', y_test.shape)

# 把訓練資料傳給 SVC 類 fit 方法來訓練演算法
print("Training started...")
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
print("Training Completed.")

# 預測新的資料類別
y_pred = svclassifier.predict(X_test)
# 計算評價指標 : 混淆矩陣、精度、召回率
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
