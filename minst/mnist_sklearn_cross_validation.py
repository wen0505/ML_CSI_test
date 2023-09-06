# 用sklearn中的SVM來訓練模型，預測數據集
from sklearn import cross_validation, svm, metrics

def load_csv(fname):
	labels=[]
	images=[]
	with open(fname,"r") as f:
		for line in f:
			cols=line.split(",")
			if len(cols)<2:continue
			labels.append(int(cols.pop(0)))
			vals=list(map(lambda n: int(n) / 256,cols))
			images.append(vals)
		return {"labels":labels,"images":images}

data=load_csv("./data/train.csv")
test=load_csv("./data/t10k.csv")

clf=svm.SVC()
clf.fit(data["images"], data["labels"])
# 訓練數據集

predict =clf.predict(test["images"])
# 預測測試集

score=metrics.accuracy_score(test["labels"], predict)
# 生成測試精度
report=metrics.classification_report(test["labels"], predict)
# 生成交叉驗證的報告
print(score)
# 顯示數據精度
print(report)
# 顯示交叉驗證數據集報告

