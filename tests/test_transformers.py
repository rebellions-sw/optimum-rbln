import os
import shutil
import unittest

import torch
from rebel.core.exception import RBLNRuntimeError
from transformers import T5EncoderModel

from optimum.rbln import (
    RBLNASTForAudioClassification,
    RBLNBartModel,
    RBLNBertForMaskedLM,
    RBLNBertForQuestionAnswering,
    RBLNCLIPTextModel,
    RBLNColPaliForRetrieval,
    RBLNDPTForDepthEstimation,
    RBLNResNetForImageClassification,
    RBLNT5EncoderModel,
    RBLNTimeSeriesTransformerForPrediction,
    RBLNWav2Vec2ForCTC,
    RBLNWhisperForConditionalGeneration,
    RBLNXLMRobertaForSequenceClassification,
    RBLNXLMRobertaModel,
)
from optimum.rbln.transformers.models.auto.modeling_auto import (
    RBLNAutoModel,
    RBLNAutoModelForAudioClassification,
    RBLNAutoModelForCTC,
    RBLNAutoModelForDepthEstimation,
    RBLNAutoModelForImageClassification,
    RBLNAutoModelForMaskedLM,
    RBLNAutoModelForQuestionAnswering,
    RBLNAutoModelForSequenceClassification,
    RBLNAutoModelForSpeechSeq2Seq,
)
from optimum.rbln.utils.runtime_utils import ContextRblnConfig
from optimum.rbln.utils.save_utils import maybe_load_preprocessors

from .test_base import BaseTest, TestLevel


RANDOM_INPUT_IDS = torch.randint(low=0, high=50, size=(1, 512), generator=torch.manual_seed(42), dtype=torch.int64)
RANDOM_ATTN_MASK = torch.randint(low=0, high=2, size=(1, 512), generator=torch.manual_seed(42), dtype=torch.int64)
RANDOM_TOKEN_TYPE_IDS = torch.randint(low=0, high=3, size=(1, 512), generator=torch.manual_seed(84), dtype=torch.int64)
RANDOM_AUDIO = torch.randn(size=(1, 160005), generator=torch.manual_seed(42), dtype=torch.float32)


class TestASTModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForAudioClassification
    RBLN_CLASS = RBLNASTForAudioClassification
    # HF_MODEL_ID = "hf-internal-testing/tiny-random-ASTForAudioClassification"

    # FIXME:: Update to internal once enabled tiny model
    HF_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
    GENERATION_KWARGS = {
        "input_values": torch.randn(size=(1, 1024, 128), generator=torch.manual_seed(42), dtype=torch.float32)
    }
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
    }


class TestResNetModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForImageClassification
    RBLN_CLASS = RBLNResNetForImageClassification
    HF_MODEL_ID = "hf-internal-testing/tiny-random-ResNetForImageClassification"
    GENERATION_KWARGS = {"pixel_values": torch.randn(1, 3, 224, 224, generator=torch.manual_seed(42))}
    TEST_LEVEL = TestLevel.ESSENTIAL

    def test_compile_from_model(self):
        if self.is_diffuser:
            pass  # TODO(diffuser from model)
        else:
            HF_CLASS = self.HF_AUTO_CLASS or self.HF_CLASS
            with ContextRblnConfig(device=self.DEVICE):
                preprocessors = maybe_load_preprocessors(self.HF_MODEL_ID)
                model = HF_CLASS.from_pretrained(
                    self.HF_MODEL_ID,
                    **self.HF_CONFIG_KWARGS,
                    **{
                        "torchscript": True,
                    },
                )
                _ = self.RBLN_CLASS.from_model(
                    model,
                    **self.RBLN_CLASS_KWARGS,
                    **self.HF_CONFIG_KWARGS,
                    preprocessors=preprocessors,  # For image_classification
                )

    def _inner_test_save_load(self, tmpdir):
        super()._inner_test_save_load(tmpdir)

        with self.assertRaises(RBLNRuntimeError):
            _ = self.RBLN_CLASS.from_pretrained(
                tmpdir,
                export=False,
                rbln_device=[0, 1, 2, 3],
            )

    def test_failed_to_create_runtime(self):
        with self.assertRaises(ValueError):
            _ = self.RBLN_CLASS.from_pretrained(
                self.HF_MODEL_ID,
                export=True,
                rbln_device=[0, 1, 2, 3],
            )


class TestBertModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForQuestionAnswering
    RBLN_CLASS = RBLNBertForQuestionAnswering
    HF_MODEL_ID = "hf-internal-testing/tiny-random-BertForQuestionAnswering"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS,
        "attention_mask": RANDOM_ATTN_MASK,
        "token_type_ids": RANDOM_TOKEN_TYPE_IDS,
    }


class TestBertForMaskedLM(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForMaskedLM
    RBLN_CLASS = RBLNBertForMaskedLM
    HF_MODEL_ID = "hf-internal-testing/tiny-random-BertForMaskedLM"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS,
        "attention_mask": RANDOM_ATTN_MASK,
        "token_type_ids": RANDOM_TOKEN_TYPE_IDS,
    }


class TestDPTModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForDepthEstimation
    RBLN_CLASS = RBLNDPTForDepthEstimation
    HF_MODEL_ID = "hf-internal-testing/tiny-random-DPTForDepthEstimation"
    GENERATION_KWARGS = {"pixel_values": torch.randn(1, 3, 32, 32, generator=torch.manual_seed(42))}


class TestT5EncoderModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = None
    RBLN_CLASS = RBLNT5EncoderModel

    HF_MODEL_ID = "t5-small"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS,
        "attention_mask": RANDOM_ATTN_MASK,
    }
    HF_CONFIG_KWARGS = {
        "num_layers": 1,
    }

    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.get_rbln_local_dir()):
            shutil.rmtree(cls.get_rbln_local_dir())

        t5_encoder_model = T5EncoderModel.from_pretrained(cls.HF_MODEL_ID, return_dict=False, **cls.HF_CONFIG_KWARGS)
        cls.model = cls.RBLN_CLASS.from_model(
            model=t5_encoder_model,
            model_save_dir=cls.get_rbln_local_dir(),
            rbln_device=-1,
            **cls.RBLN_CLASS_KWARGS,
        )


class TestWhisperModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForSpeechSeq2Seq
    RBLN_CLASS = RBLNWhisperForConditionalGeneration
    HF_MODEL_ID = "openai/whisper-tiny"
    DEVICE = 0

    GENERATION_KWARGS = {
        "input_features": torch.randint(
            low=-1, high=1, size=(2, 80, 3000), generator=torch.manual_seed(42), dtype=torch.float32
        ),
        "max_new_tokens": 10,
    }
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 4,
        "encoder_layers": 4,
        "decoder_layers": 4,
    }
    RBLN_CLASS_KWARGS = {
        "rbln_token_timestamps": False,
        "rbln_batch_size": 2,
    }

    def test_generate(self):
        from datasets import load_dataset
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.HF_MODEL_ID)
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True
        )
        input_features_list = []

        for i in range(2):
            input_features = processor(
                ds[i]["audio"]["array"],
                sampling_rate=ds[i]["audio"]["sampling_rate"],
                truncation=False,
                return_tensors="pt",
            ).input_features
            input_features_list.append(input_features)
        input_features = torch.cat(input_features_list, dim=0)
        output = self.model.generate(input_features=input_features, max_new_tokens=10)
        self.EXPECTED_OUTPUT = output[:, :5]

        # test_generate_language
        output = self.model.generate(input_features=input_features, max_new_tokens=10, language="en")[:, :5]
        self.assertTrue(torch.all(output == self.EXPECTED_OUTPUT))

        # test_generate_language_auto_detect
        output = self.model.generate(input_features=input_features, max_new_tokens=10, language=None)[:, :5]
        self.assertTrue(torch.all(output == self.EXPECTED_OUTPUT))

    def test_long_form_language_auto_detect_generate(self):
        inputs = self.get_inputs()

        inputs["input_features"] = torch.randint(
            low=-1, high=1, size=(2, 80, 3001), generator=torch.manual_seed(42), dtype=torch.float32
        )
        inputs["attention_mask"] = torch.ones(2, 3002, dtype=torch.int64)

        _ = self.model.generate(**inputs, temperature=0.0, language=None, return_timestamps=True)

    def test_long_form_generate(self):
        inputs = self.get_inputs()

        inputs["input_features"] = torch.randint(
            low=-1, high=1, size=(2, 80, 3001), generator=torch.manual_seed(42), dtype=torch.float32
        )
        inputs["attention_mask"] = torch.ones(2, 3002, dtype=torch.int64)

        _ = self.model.generate(**inputs, temperature=0.0, return_timestamps=True)

    def test_pipeline(self):
        import numpy as np
        from transformers import AutoProcessor, pipeline

        processor = AutoProcessor.from_pretrained(self.HF_MODEL_ID)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            return_timestamps=True,
            batch_size=2,
        )

        data = [np.random.rand(5000), np.random.rand(5000)]

        with torch.no_grad():
            _ = pipe(
                data,
                generate_kwargs={
                    "repetition_penalty": 1.3,
                    "num_beams": 1,
                },
                batch_size=2,
            )


class TestWhisperModel_TokenTimestamps(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForSpeechSeq2Seq
    RBLN_CLASS = RBLNWhisperForConditionalGeneration
    HF_MODEL_ID = "openai/whisper-tiny"

    GENERATION_KWARGS = {
        "input_features": torch.randint(
            low=0, high=50, size=(2, 80, 3000), generator=torch.manual_seed(42), dtype=torch.float32
        ),
        "max_new_tokens": 1,
        "return_token_timestamps": True,
        "return_timestamps": True,
        "generation_config": None,
    }

    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
        "encoder_layers": 1,
        "decoder_layers": 4,
    }

    RBLN_CLASS_KWARGS = {
        "rbln_token_timestamps": True,
        "rbln_batch_size": 2,
    }

    def test_generate(self):
        inputs = self.get_inputs()
        _ = self.model.generate(**inputs)

    def test_long_form_generate(self):
        inputs = self.get_inputs()

        inputs["input_features"] = torch.randint(
            low=0, high=50, size=(2, 80, 3001), generator=torch.manual_seed(42), dtype=torch.float32
        )
        inputs["attention_mask"] = torch.ones(2, 3002, dtype=torch.int64)

        _ = self.model.generate(**inputs, temperature=0.0)


class TestRBLNXLMRobertaForSequenceClassification(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForSequenceClassification
    RBLN_CLASS = RBLNXLMRobertaForSequenceClassification

    # FIXME:: Update to internal once enabled tiny model
    HF_MODEL_ID = "BAAI/bge-reranker-v2-m3"
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 128}
    GENERATION_KWARGS = {
        "input_ids": torch.randint(low=0, high=50, size=(1, 128), generator=torch.manual_seed(42), dtype=torch.int64),
        "attention_mask": torch.randint(
            low=0, high=2, size=(1, 128), generator=torch.manual_seed(42), dtype=torch.int64
        ),
    }
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
    }


class TestXLMRobertaModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModel
    RBLN_CLASS = RBLNXLMRobertaModel
    TEST_LEVEL = TestLevel.FULL
    # HF_MODEL_ID = "hf-internal-testing/tiny-xlm-roberta"

    # FIXME:: Update to internal once enabled tiny model
    HF_MODEL_ID = "BAAI/bge-m3"
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 128}
    GENERATION_KWARGS = {
        "input_ids": torch.randint(low=0, high=50, size=(1, 128), generator=torch.manual_seed(42), dtype=torch.int64),
        "attention_mask": torch.randint(
            low=0, high=2, size=(1, 128), generator=torch.manual_seed(42), dtype=torch.int64
        ),
    }
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
        "vocab_size": 1024,
        "ignore_mismatched_sizes": True,
    }


class TestCLIPModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNCLIPTextModel
    HF_MODEL_ID = "hf-internal-testing/tiny-random-CLIPModel"
    GENERATION_KWARGS = {
        "input_ids": RANDOM_INPUT_IDS,
        "attention_mask": RANDOM_ATTN_MASK,
    }


class TestColPaliModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNColPaliForRetrieval
    HF_MODEL_ID = "thkim93/colpali-hf-1layer"
    GENERATION_KWARGS = {
        "input_ids": torch.full((1, 1024), fill_value=257152, dtype=torch.int32),
        "attention_mask": torch.ones((1, 1024), dtype=torch.int32),
        "pixel_values": torch.randn(1, 3, 448, 448, generator=torch.manual_seed(42)),
    }


class TestWav2VecModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForCTC
    RBLN_CLASS = RBLNWav2Vec2ForCTC
    HF_MODEL_ID = "hf-internal-testing/tiny-random-Wav2Vec2ForCTC"
    GENERATION_KWARGS = {"input_values": RANDOM_AUDIO}
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 160005}


class TestTimeSeriesTransformerForPrediction(BaseTest.TestModel):
    RBLN_AUTO_CLASS = None
    RBLN_CLASS = RBLNTimeSeriesTransformerForPrediction
    HF_MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
    GENERATION_KWARGS = {
        "static_categorical_features": torch.ones(1, 1, dtype=torch.long),
        "static_real_features": torch.ones(1, 1, dtype=torch.float),
        "past_time_features": torch.randn(1, 61, 2),
        "past_values": torch.randn(1, 61),
        "past_observed_mask": torch.ones(1, 61, dtype=torch.long),
        "future_time_features": torch.randn(1, 24, 2),
    }
    RBLN_CLASS_KWARGS = {"rbln_batch_size": 1, "rbln_num_parallel_samples": 100}
    DEVICE = 0

    def test_generate(self):
        inputs = self.get_inputs()
        _ = self.model.generate(**inputs)


class TestRBLNBartModel(BaseTest.TestModel):
    RBLN_CLASS = RBLNBartModel
    HF_MODEL_ID = "hf-internal-testing/tiny-random-BartModel"
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 100}
    GENERATION_KWARGS = {
        "input_ids": torch.randint(low=0, high=50, size=(1, 100), generator=torch.manual_seed(42), dtype=torch.int64),
        "attention_mask": torch.randint(
            low=0, high=2, size=(1, 100), generator=torch.manual_seed(42), dtype=torch.int64
        ),
    }


if __name__ == "__main__":
    unittest.main()
