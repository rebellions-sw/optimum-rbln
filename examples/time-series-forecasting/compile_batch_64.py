from optimum.rbln import RBLNTimeSeriesTransformerForPrediction

model_id = "huggingface/time-series-transformer-tourism-monthly"

rbln_model = RBLNTimeSeriesTransformerForPrediction.from_pretrained(
    model_id,
    export=True,  
    rbln_batch_size=64
)