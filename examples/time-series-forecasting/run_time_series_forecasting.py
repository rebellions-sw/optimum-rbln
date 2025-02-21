import torch
from huggingface_hub import hf_hub_download
from transformers import TimeSeriesTransformerForPrediction

from optimum.rbln import RBLNTimeSeriesTransformerForPrediction

rbln_model = RBLNTimeSeriesTransformerForPrediction.from_pretrained("huggingface/time-series-transformer-tourism-monthly",export=True)

# 데이터 다운로드 및 로드
file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

# 사전 학습된 Time Series Transformer 모델 로드
model = TimeSeriesTransformerForPrediction.from_pretrained("huggingface/time-series-transformer-tourism-monthly")



# 예측 수행
outputs = model.generate(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"],
    static_real_features=batch["static_real_features"],
    future_time_features=batch["future_time_features"],
)

# 평균 예측값 계산
mean_prediction = outputs.sequences.mean(dim=1)

print(mean_prediction)

past_observed_mask = torch.nn.functional.pad(batch["past_observed_mask"], (0, 3))
past_values = torch.nn.functional.pad(batch["past_values"], (0, 3))
past_time_features = torch.nn.functional.pad(batch["past_time_features"], (0, 0, 0, 3))

outputs = model.generate(
    past_values=past_values,
    past_time_features=past_time_features,
    past_observed_mask=past_observed_mask,
    static_categorical_features=batch["static_categorical_features"],
    static_real_features=batch["static_real_features"],
    future_time_features=batch["future_time_features"],
)

# 평균 예측값 계산
mean_prediction = outputs.sequences.mean(dim=1)

# 결과 출력
print(mean_prediction)
