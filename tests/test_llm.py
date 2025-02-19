import json
import unittest
import warnings

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from optimum.rbln import (
    RBLNAutoModel,
    RBLNAutoModelForCausalLM,
    RBLNAutoModelForSeq2SeqLM,
    RBLNAutoModelForVision2Seq,
    RBLNBartForConditionalGeneration,
    RBLNExaoneForCausalLM,
    RBLNGPT2LMHeadModel,
    RBLNLlamaForCausalLM,
    RBLNLlavaNextForConditionalGeneration,
    RBLNPhiForCausalLM,
    RBLNQwen2ForCausalLM,
    RBLNT5ForConditionalGeneration,
)

from .test_base import BaseTest, DisallowedTestBase, TestLevel


RANDOM_ATTN_MASK = torch.randint(low=0, high=2, size=(1, 512), generator=torch.manual_seed(42), dtype=torch.int64)
RANDOM_TOKEN_TYPE_IDS = torch.randint(low=0, high=3, size=(1, 512), generator=torch.manual_seed(84), dtype=torch.int64)
RANDOM_INPUT_FEATURES = torch.randint(
    low=0, high=50, size=(1, 80, 3000), generator=torch.manual_seed(42), dtype=torch.float32
)


class LLMTest:
    class TestLLM(BaseTest.TestModel):
        _tokenizer = None
        RBLN_AUTO_CLASS = RBLNAutoModelForCausalLM
        DEVICE = None  # Use device to run
        PROMPT = "Who are you?"

        @classmethod
        @property
        def tokenizer(cls):
            if cls._tokenizer is None:
                cls._tokenizer = AutoTokenizer.from_pretrained(cls.HF_MODEL_ID)
            return cls._tokenizer

        def get_inputs(self):
            inputs = self.tokenizer(self.PROMPT, return_tensors="pt")
            inputs["max_new_tokens"] = 20
            inputs["do_sample"] = False
            return inputs

        def postprocess(self, inputs, output):
            input_len = inputs["input_ids"].shape[-1]
            generated_text = self.tokenizer.decode(
                output[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return generated_text


class TestQwen2Model(LLMTest.TestLLM):
    RBLN_CLASS = RBLNQwen2ForCausalLM
    HF_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
    EXPECTED_OUTPUT = " I am a 30-year-old woman who has been living with lupus for over 1"
    HF_CONFIG_KWARGS = {"max_position_embeddings": 1024}


class TestLlamaForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    TEST_LEVEL = TestLevel.ESSENTIAL
    EXPECTED_OUTPUT = "reress makefable R���� noethetsshss rechoolso�"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}

    def get_inputs(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(self.PROMPT, return_tensors="pt")
        return inputs


class TestLlamaForCausalLM_Flash(LLMTest.TestLLM):
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    TEST_LEVEL = TestLevel.ESSENTIAL
    EXPECTED_OUTPUT = "reress makefable R���� noethetsshss rechoolso�"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 8192}
    RBLN_CLASS_KWARGS = {"rbln_config": {"attn_impl": "flash_attn", "kvcache_partition_len": 4096}}

    def get_inputs(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(self.PROMPT, return_tensors="pt")
        return inputs


class TestGPT2LMHeadModel(LLMTest.TestLLM):
    RBLN_CLASS = RBLNGPT2LMHeadModel
    # TEST_LEVEL = TestLevel.FULL
    EXPECTED_OUTPUT = (
        " What kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind"
    )
    HF_MODEL_ID = "openai-community/gpt2"
    HF_CONFIG_KWARGS = {"n_layer": 1, "max_position_embeddings": 1024}


class TestPhiForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNPhiForCausalLM

    # HF_MODEL_ID = "hf-internal-testing/tiny-random-PhiForCausalLM"
    HF_MODEL_ID = "microsoft/phi-2"
    EXPECTED_OUTPUT = "\nAnswer: Theorettebrates']['<<<urlskolegateezzingrill"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024, "trust_remote_code": True}


class TestExaoneForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNExaoneForCausalLM
    # HF_MODEL_ID = "katuni4ka/tiny-random-exaone"
    HF_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    EXPECTED_OUTPUT = "????????????????????"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024, "trust_remote_code": True}


class TestT5Model(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForSeq2SeqLM
    RBLN_CLASS = RBLNT5ForConditionalGeneration

    # HF_MODEL_ID = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
    # FIXME:: Update to internal once enabled tiny model
    HF_MODEL_ID = "t5-small"
    PROMPT = "summarize: studies have shown that owning a dog is good for you"
    EXPECTED_OUTPUT = ""
    RBLN_CLASS_KWARGS = {"rbln_config": {"enc_max_seq_len": 512, "dec_max_seq_len": 512}}
    HF_CONFIG_KWARGS = {"num_layers": 1}

    def get_inputs(self):
        inputs = self.tokenizer(
            self.PROMPT, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        inputs["num_beams"] = 1
        return inputs

    def postprocess(self, inputs, output):
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return generated_text


class TestBartModel(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForSeq2SeqLM
    RBLN_CLASS = RBLNBartForConditionalGeneration

    # HF_MODEL_ID = "sshleifer/bart-tiny-random"
    # FIXME:: Update to internal once enabled tiny model
    HF_MODEL_ID = "lucadiliello/bart-small"
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
        "decoder_layers": 1,
        "encoder_layers": 1,
    }
    RBLN_CLASS_KWARGS = {"rbln_config": {"enc_max_seq_len": 512, "dec_max_seq_len": 512}}
    PROMPT = "summarize: studies have shown that owning a dog is good for you"
    EXPECTED_OUTPUT = "InsteadInsteadInsteadHoweverHoweverHoweverAlthoughAlthoughAlthoughWhileWhileWhileAlthoughAlthoughHoweverHoweverManyMany"
    TEST_LEVEL = TestLevel.ESSENTIAL

    def get_inputs(self):
        inputs = self.tokenizer(
            self.PROMPT, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        inputs["num_beams"] = 1
        return inputs

    def postprocess(self, inputs, output):
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return generated_text

    def test_automap(self):
        # BartForConditionalGeneration -> RBLNBartForConditionalGeneration compile case
        with self.subTest():
            assert self.RBLN_CLASS == self.RBLN_AUTO_CLASS.get_rbln_cls(
                self.HF_MODEL_ID,
                **self.RBLN_CLASS_KWARGS,
                **self.HF_CONFIG_KWARGS,
            )

        # BartForConditionalGeneration -> RBLNBartModel compile case
        # Invoked rbln_class is different from config's architecture
        with self.subTest():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                RBLNAutoModel.get_rbln_cls(self.HF_MODEL_ID)

                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertIn("This mismatch could cause some operations", str(w[-1].message))

        # BartForConditionalGeneration -> RBLNBartForCausalLM compile case
        # RBLNBartForCausalLM is not yet supported in optimum.rbln
        with self.subTest():
            with pytest.raises(AttributeError):
                RBLNAutoModelForCausalLM.get_rbln_cls(self.HF_MODEL_ID)

        # RBLNBartForSeq2SeqLM -> RBLNBartForCausalLM load case
        with self.subTest():
            with pytest.raises(ValueError):
                _ = RBLNAutoModelForCausalLM.from_pretrained(
                    self.RBLN_LOCAL_DIR,
                    export=False,
                    rbln_create_runtimes=False,
                    **self.HF_CONFIG_KWARGS,
                )


class TestLlavaNextForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNLlavaNextForConditionalGeneration
    TEST_LEVEL = TestLevel.FULL
    HF_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"  # No tiny model yet.
    LLAVA_PIXEL_VALUES = torch.randn(1, 5, 3, 336, 336, generator=torch.manual_seed(42))
    LLAVA_IMAGE_SIZES = torch.tensor([[900, 900]], dtype=torch.int64)
    GENERATION_KWARGS = {
        "input_ids": torch.randint(
            low=0,
            high=50,
            size=(1, 512),
            generator=torch.manual_seed(42),
            dtype=torch.int64,
        ),
        "attention_mask": torch.randint(
            low=0, high=2, size=(1, 512), generator=torch.manual_seed(42), dtype=torch.int64
        ),
        "pixel_values": LLAVA_PIXEL_VALUES,
        "image_sizes": LLAVA_IMAGE_SIZES,
        "max_new_tokens": 10,
    }
    RBLN_CLASS_KWARGS = {"rbln_config": {"language_model": {"use_inputs_embeds": True}}}
    EXPECTED_OUTPUT = "ẩ kennisSoft biologieussyózdsi fraulent data"

    # override
    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID)

        # To test image input, input_id should contain special image token.
        cls.GENERATION_KWARGS["input_ids"][0, 44] = config.image_token_index
        text_config = json.loads(config.text_config.to_json_string())
        text_config["num_hidden_layers"] = 1
        kwargs = {"text_config": text_config}
        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    def get_inputs(self):
        return self.GENERATION_KWARGS


class TestDisallowedLlama_1(DisallowedTestBase.DisallowedTest):
    # Too long sequence length
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 32768 * 2}


class TestDisallowedLlama_2(DisallowedTestBase.DisallowedTest):
    # Flash attn : Not multiple
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 8192}
    RBLN_CLASS_KWARGS = {"rbln_config": {"attn_impl": "flash_attn", "kvcache_partition_len": 8000}}


class TestDisallowedLlama_3(DisallowedTestBase.DisallowedTest):
    # Flash attn : too short partition
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 8192}
    RBLN_CLASS_KWARGS = {"rbln_config": {"attn_impl": "flash_attn", "kvcache_partition_len": 1024}}


class TestDisallowedLlama_4(DisallowedTestBase.DisallowedTest):
    # Flash attn : too short max_seq_len
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 2048}
    RBLN_CLASS_KWARGS = {"rbln_config": {"attn_impl": "flash_attn", "kvcache_partition_len": 1024}}


if __name__ == "__main__":
    unittest.main()
