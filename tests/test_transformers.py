import os
import shutil
import unittest

import torch
from transformers import T5EncoderModel

from optimum.rbln import (
    RBLNASTForAudioClassification,
    RBLNBertForQuestionAnswering,
    RBLNCLIPTextModel,
    RBLNDPTForDepthEstimation,
    RBLNResNetForImageClassification,
    RBLNT5EncoderModel,
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
RANDOM_INPUT_FEATURES = torch.randint(
    low=0, high=50, size=(1, 80, 3000), generator=torch.manual_seed(42), dtype=torch.float32
)


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
                        "return_dict": False,
                    },
                )
                _ = self.RBLN_CLASS.from_model(
                    model,
                    **self.RBLN_CLASS_KWARGS,
                    **self.HF_CONFIG_KWARGS,
                    preprocessors=preprocessors,  # For image_classification
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
        if os.path.exists(cls.RBLN_LOCAL_DIR):
            shutil.rmtree(cls.RBLN_LOCAL_DIR)

        t5_encoder_model = T5EncoderModel.from_pretrained(cls.HF_MODEL_ID, return_dict=False, **cls.HF_CONFIG_KWARGS)
        cls.model = cls.RBLN_CLASS.from_model(
            model=t5_encoder_model,
            model_save_dir=cls.RBLN_LOCAL_DIR,
            rbln_device=-1,
            **cls.RBLN_CLASS_KWARGS,
        )


class TestWhisperModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForSpeechSeq2Seq
    RBLN_CLASS = RBLNWhisperForConditionalGeneration
    HF_MODEL_ID = "openai/whisper-tiny"

    GENERATION_KWARGS = {
        "input_features": RANDOM_INPUT_FEATURES,
        "max_new_tokens": 10,
    }
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
        "encoder_layers": 1,
        "decoder_layers": 1,
    }


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


class TestWav2VecModel(BaseTest.TestModel):
    RBLN_AUTO_CLASS = RBLNAutoModelForCTC
    RBLN_CLASS = RBLNWav2Vec2ForCTC
    HF_MODEL_ID = "hf-internal-testing/tiny-random-Wav2Vec2ForCTC"
    GENERATION_KWARGS = {"input_values": RANDOM_AUDIO}
    RBLN_CLASS_KWARGS = {"rbln_max_seq_len": 160005}


if __name__ == "__main__":
    unittest.main()
