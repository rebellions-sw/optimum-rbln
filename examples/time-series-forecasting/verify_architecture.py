import torch
from huggingface_hub import hf_hub_download
from transformers import TimeSeriesTransformerForPrediction

from optimum.rbln import RBLNTimeSeriesTransformerForPrediction
from optimum.rbln.transformers.models.time_series_transformers.time_series_transformers_architecture import (
    TimeSeriesTransformersWrapper,
)


# 데이터 다운로드 및 로드
file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

rbln_model = RBLNTimeSeriesTransformerForPrediction.from_pretrained("huggingface/time-series-transformer-tourism-monthly",export=True)
# 사전 학습된 Time Series Transformer 모델 로드
torch_model = TimeSeriesTransformerForPrediction.from_pretrained("huggingface/time-series-transformer-tourism-monthly")


wrapped_model = TimeSeriesTransformersWrapper(torch_model)

enc_inputs_embeds = torch.load("enc_inputs.pt")
enc_last_hidden_states = torch.load("enc_last_hidden_state.pt")

dec_inputs_embeds = torch.load("decoder_input_embed.pt")
dec_past_key_values = torch.load("dec_past_key_values.pt")
dec_last_hidden_states = torch.load("dec_last_hidden_state.pt")

with torch.no_grad():
    cross_key_values = torch.zeros(4, 64, 2, 24, 13)
    cross_key_values = wrapped_model.encoder(enc_inputs_embeds, cross_key_values)

    self_past_kv = [torch.zeros(6400, 2, 24, 13) for _ in range(4)]
    decoder_attention_mask = torch.zeros(1, 24)
    decoder_attention_mask[:, 0] = 1

    rbln_dec_output = wrapped_model.decoder(
        dec_inputs_embeds, decoder_attention_mask, torch.tensor(0), cross_key_values, *self_past_kv
    )

print(rbln_dec_output[0] - dec_last_hidden_states)
breakpoint()


# past_observed_mask = torch.nn.functional.pad(batch["past_observed_mask"], (0, 3))
# past_values = torch.nn.functional.pad(batch["past_values"], (0, 3))
# past_time_features = torch.nn.functional.pad(batch["past_time_features"], (0, 0, 0, 3))

# outputs = model.generate(
#     past_values=past_values,
#     past_time_features=past_time_features,
#     past_observed_mask=past_observed_mask,
#     static_categorical_features=batch["static_categorical_features"],
#     static_real_features=batch["static_real_features"],
#     future_time_features=batch["future_time_features"],
# )

# # 평균 예측값 계산
# mean_prediction = outputs.sequences.mean(dim=1)

# # 결과 출력
# print(mean_prediction)
