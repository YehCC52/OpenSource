# train.py
from sklearn import svm, datasets
import bentoml

# 載入 iris 資料集
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 建立並訓練模型
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# 儲存模型到 BentoML 模型倉庫
bentoml.sklearn.save_model("iris_clf", clf)

