import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# (1) 讀取 csv 檔
PATH = 'C:/Users/lab517/PycharmProjects/svm_csi/src_SVM/amp_stand_sit.csv'
# 先標示出需要使用的 column
usecol0 = ['group', 'index#1:', 'index#2:', 'index#3:', 'index#4:', 'index#5:', 'index#6:', 'index#7:', 'index#8:',
           'index#9:', 'index#10:', 'index#11:', 'index#12:', 'index#13:', 'index#14:', 'index#15:', 'index#16:',
           'index#17:', 'index#18:', 'index#19:', 'index#20:', 'index#21:', 'index#22:', 'index#23:', 'index#24:',
           'index#25:', 'index#26:', 'index#27:', 'index#28:', 'index#29:', 'index#30:', 'index#31:', 'index#32:',
           'index#33:', 'index#34:', 'index#35:', 'index#36:', 'index#37:', 'index#38:', 'index#39:', 'index#40:',
           'index#41:', 'index#42:', 'index#43:', 'index#44:', 'index#45:', 'index#46:', 'index#47:', 'index#48:',
           'index#49:', 'index#50:', 'index#51:', 'index#52:']

df = pd.read_csv(PATH)
df_data = pd.DataFrame(data=df, columns=usecol0)
# print(df_data)
# -------------------------------------------------------------------
# 未執行 : (2) 探索數據找出what they look like
#         (3) 預處理數據 (由 write_amp_list.py 執行)
# -------------------------------------------------------------------
# (4) 把屬性和類別標籤分開
X = df_data.drop(labels=['group'], axis=1).values       # 移除 group 並取得剩下欄位資料
y = df_data['group'].values
# print(X)
# -------------------------------------------------------------------
# (5) 切割訓練集與測試集
print("Dividing the dataset into testing and training sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('train shape:', X_train.shape)
print('test shape:', X_test.shape)
print('train shape:', y_train.shape)
print('test shape:', y_test.shape)
# -------------------------------------------------------------------
# (6) 把訓練資料傳給 SVC 類 fit 方法來訓練演算法
print("Training started...")
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
print("Training Completed.")
# -------------------------------------------------------------------
# (7) 預測新的資料類別
y_pred = svclassifier.predict(X_test)
# -------------------------------------------------------------------
# (8) 計算評價指標 : 混淆矩陣、精度、召回率
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
