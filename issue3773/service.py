# service.py
import bentoml
from bentoml.io import JSON
import pandas as pd

# 載入模型 runner
model_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

# 建立 BentoML 服務
svc = bentoml.Service("iris_classifier_service", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def predict(input_json):
    df = pd.DataFrame(input_json)
    result = model_runner.predict.run(df)
    return result.tolist()

