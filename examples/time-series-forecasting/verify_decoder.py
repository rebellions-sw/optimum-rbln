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

# 사전 학습된 Time Series Transformer 모델 로드
rbln_model = RBLNTimeSeriesTransformerForPrediction.from_pretrained("time-series-transformer-tourism-monthly",export=False)
torch_model = TimeSeriesTransformerForPrediction.from_pretrained("huggingface/time-series-transformer-tourism-monthly")


wrapped_model = TimeSeriesTransformersWrapper(torch_model)


for i in range(64):

    enc_inputs_embeds = torch.load("enc_inputs.pt")[i:i+1]
    enc_last_hidden_states = torch.load("enc_last_hidden_state.pt")[i:i+1].repeat_interleave(100, dim=0)
    dec_inputs_embeds = torch.load("decoder_input_embed.pt")[i*100:i*100+100]
    dec_past_key_values = torch.load("dec_past_key_values.pt")
    dec_last_hidden_states = torch.load("dec_last_hidden_state.pt")[i*100:i*100+100]

    with torch.no_grad():
        # 1. golden
        golden_output = torch_model.get_decoder()(inputs_embeds = dec_inputs_embeds, encoder_hidden_states = enc_last_hidden_states).last_hidden_state
        
        # 2. Wrapped
        cross_key_values = torch.zeros(4, 1, 2, 24, 13)
        cross_key_values = wrapped_model.encoder(enc_inputs_embeds, cross_key_values)
        self_past_kv = [torch.zeros(100, 2, 24, 13) for _ in range(4)]
        decoder_attention_mask = torch.zeros(100, 24)
        decoder_attention_mask[:, 0] = 1
        padded_dec_inputs_embeds = torch.nn.functional.pad(
            dec_inputs_embeds, (0, 0, 0, 23)
        )
        wrapped_dec_output = wrapped_model.decoder(
            padded_dec_inputs_embeds, decoder_attention_mask, torch.tensor(0), cross_key_values, *self_past_kv
        )[0][:,:1]
        
        # 3. rbln
        rbln_dec_output = rbln_model.decoder(
            padded_dec_inputs_embeds, decoder_attention_mask, torch.tensor(0,dtype=torch.int32)
        )[:,:1]
        
    print(f"batch {i}")
    print("wrap vs golden")
    print((wrapped_dec_output - golden_output).abs().max())

    print("rbln vs golden")
    print((rbln_dec_output- golden_output).abs().max())

    from scipy import stats

    res = stats.pearsonr(rbln_dec_output[0][0], golden_output[0][0]).statistic
    print(f"pearsonr: {res.item()}")

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
