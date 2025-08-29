import json
import os
import unittest
import warnings

import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from optimum.rbln import (
    RBLNAutoModel,
    RBLNAutoModelForCausalLM,
    RBLNAutoModelForImageTextToText,
    RBLNAutoModelForSeq2SeqLM,
    RBLNAutoModelForVision2Seq,
    RBLNBartForConditionalGeneration,
    RBLNBlip2ForConditionalGeneration,
    RBLNExaoneForCausalLM,
    RBLNGemma3ForCausalLM,
    RBLNGemma3ForConditionalGeneration,
    RBLNGPT2LMHeadModel,
    RBLNGPT2Model,
    RBLNIdefics3ForConditionalGeneration,
    RBLNLlamaForCausalLM,
    RBLNLlamaModel,
    RBLNLlavaForConditionalGeneration,
    RBLNLlavaNextForConditionalGeneration,
    RBLNMistralForCausalLM,
    RBLNMistralModel,
    RBLNOPTForCausalLM,
    RBLNOPTModel,
    RBLNPegasusForConditionalGeneration,
    RBLNPhiForCausalLM,
    RBLNPhiModel,
    RBLNQwen2_5_VLForConditionalGeneration,
    RBLNQwen2ForCausalLM,
    RBLNQwen2Model,
    RBLNQwen3ForCausalLM,
    RBLNQwen3Model,
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
        def get_tokenizer(cls):
            if cls._tokenizer is None:
                cls._tokenizer = AutoTokenizer.from_pretrained(cls.HF_MODEL_ID)
            return cls._tokenizer

        def get_inputs(self):
            inputs = self.get_tokenizer()(self.PROMPT, return_tensors="pt")
            if self.model.can_generate():
                inputs["max_new_tokens"] = 20
                inputs["do_sample"] = False
            return inputs

        def postprocess(self, inputs, output):
            input_len = inputs["input_ids"].shape[-1]
            generated_text = self.get_tokenizer().decode(
                output[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return generated_text

    class TestLLMWithoutLMHead(TestLLM):
        RBLN_AUTO_CLASS = RBLNAutoModel


class TestMistralForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNMistralForCausalLM
    HF_MODEL_ID = "openaccess-ai-collective/tiny-mistral"
    EXPECTED_OUTPUT = "watasurescid completionennen Brad completion жеULT ba completion影 Fin сво Regimentixon cabin影 provisions bland"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024, "sliding_window": 512}


class TestMistralModel(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNMistralModel
    HF_MODEL_ID = "openaccess-ai-collective/tiny-mistral"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024, "sliding_window": 512}


class TestQwen2ForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNQwen2ForCausalLM
    HF_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
    EXPECTED_OUTPUT = "?:雨成名ylonclaimer淡elsinki一角一角一角一角一角一角一角一角一角一角一角一角一角"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}


class TestQwen2Model(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNQwen2Model
    HF_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}


class TestQwen3ForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNQwen3ForCausalLM
    HF_MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    EXPECTED_OUTPUT = (
        "יל synd Fitz Fitz Fitz Fitz Fitz Fitz Fitz Fitz Fitz Fitz Fitz Fitz Fitz Fitz_inventory天河 sanitary中途"
    )
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}


class TestQwen3ForCausalLM_UAM(TestQwen3ForCausalLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForCausalLM
    RBLN_CLASS_KWARGS = {"rbln_config": {"use_attention_mask": True}}


class TestQwen3Model(LLMTest.TestLLMWithoutLMHead):
    RBLN_AUTO_CLASS = RBLNAutoModel
    RBLN_CLASS = RBLNQwen3Model
    HF_MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}


class TestQwen3Model_UAM(TestQwen3Model):
    RBLN_CLASS_KWARGS = {"rbln_config": {"use_attention_mask": True}}


class TestOPTForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNOPTForCausalLM
    HF_MODEL_ID = "facebook/opt-2.7b"
    EXPECTED_OUTPUT = "????,,,,,,,,,,,,,,,,"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 2048}


class TestOPTModel(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNOPTModel
    HF_MODEL_ID = "facebook/opt-2.7b"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 2048}


class TestLlamaForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    TEST_LEVEL = TestLevel.ESSENTIAL
    EXPECTED_OUTPUT = "reress makefable R���� noethetsshss rechoolso�"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}

    def get_inputs(self):
        self.get_tokenizer().pad_token = self.get_tokenizer().eos_token
        inputs = self.get_tokenizer()(self.PROMPT, return_tensors="pt")
        return inputs


class TestLlamaModel(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNLlamaModel
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024}


class TestLlamaForCausalLM_Flash(LLMTest.TestLLM):
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    TEST_LEVEL = TestLevel.ESSENTIAL
    EXPECTED_OUTPUT = "reress makefable R���� noethetsshss rechoolso�"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 8192}
    RBLN_CLASS_KWARGS = {"rbln_config": {"attn_impl": "flash_attn", "kvcache_partition_len": 4096}}

    def get_inputs(self):
        self.get_tokenizer().pad_token = self.get_tokenizer().eos_token
        inputs = self.get_tokenizer()(self.PROMPT, return_tensors="pt")
        return inputs


class TestLlamaModel_Flash(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNLlamaModel
    HF_MODEL_ID = "afmck/testing-llama-tiny"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 8192}
    RBLN_CLASS_KWARGS = {"rbln_config": {"attn_impl": "flash_attn", "kvcache_partition_len": 4096}}


class TestLlamaForCausalLM_Multibatch(TestLlamaForCausalLM):
    PROMPT = ["Who are you?", "What is the capital of France?", "What is the capital of Germany?"]
    EXPECTED_OUTPUT = [
        "reress makefable R���� noethetss0oss invetetet",
        "resget makeget makeichget makeichualichual#choolchool accngngngng",
        "resget makeget makeichget makeichualichual#choolchool accngngngng",
    ]
    RBLN_CLASS_KWARGS = {"rbln_config": {"batch_size": 3, "decoder_batch_sizes": [3, 2, 1]}}

    def get_inputs(self):
        self.get_tokenizer().pad_token = self.get_tokenizer().eos_token
        inputs = self.get_tokenizer()(self.PROMPT, return_tensors="pt", padding=True)
        return inputs

    def postprocess(self, inputs, output):
        generated_texts = []
        for i in range(inputs["input_ids"].shape[0]):
            input_len = inputs["input_ids"].shape[-1]
            generated_text = self.get_tokenizer().decode(
                output[i][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            generated_texts.append(generated_text)
        return generated_texts


class TestGPT2LMHeadModel(LLMTest.TestLLM):
    RBLN_CLASS = RBLNGPT2LMHeadModel
    EXPECTED_OUTPUT = (
        " What kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind"
    )
    HF_MODEL_ID = "openai-community/gpt2"
    HF_CONFIG_KWARGS = {"n_layer": 1, "max_position_embeddings": 1024}


class TestGPT2Model(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNGPT2Model
    HF_MODEL_ID = "openai-community/gpt2"
    HF_CONFIG_KWARGS = {"n_layer": 1, "max_position_embeddings": 1024}


class TestPhiForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNPhiForCausalLM

    # HF_MODEL_ID = "hf-internal-testing/tiny-random-PhiForCausalLM"
    HF_MODEL_ID = "microsoft/phi-2"
    EXPECTED_OUTPUT = "\nAnswer: Theorettebrates']['<<<urlskolegateezzingrill"
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1, "max_position_embeddings": 1024, "trust_remote_code": True}


class TestPhiModel(LLMTest.TestLLMWithoutLMHead):
    RBLN_CLASS = RBLNPhiModel
    HF_MODEL_ID = "microsoft/phi-2"
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
        inputs = self.get_tokenizer()(
            self.PROMPT, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        inputs["num_beams"] = 1
        return inputs

    def postprocess(self, inputs, output):
        generated_text = self.get_tokenizer().decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
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
        inputs = self.get_tokenizer()(
            self.PROMPT, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        inputs["num_beams"] = 1
        return inputs

    def postprocess(self, inputs, output):
        generated_text = self.get_tokenizer().decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
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
                    self.get_rbln_local_dir(),
                    export=False,
                    rbln_create_runtimes=False,
                    **self.HF_CONFIG_KWARGS,
                )


class TestLlavaForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNLlavaForConditionalGeneration
    HF_MODEL_ID = "trl-internal-testing/tiny-LlavaForConditionalGeneration"
    PROMPT = "[INST] <image>\nWhat’s shown in this image? [/INST]"
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "vision_tower": {"output_hidden_states": True},
            "language_model": {"use_inputs_embeds": True},
        }
    }
    EXPECTED_OUTPUT = "ambbrow nur Well chimCore rapideraine Йye questaédédates Ken neu Airport din termeächstthread"
    HF_CONFIG_KWARGS = {"revision": "8ab8bfc820a6bb9e0f8de1ac715f4b53db44e684"}

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoProcessor.from_pretrained(cls.HF_MODEL_ID, revision=cls.HF_CONFIG_KWARGS["revision"])
        return cls._tokenizer

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        inputs = tokenizer(images=[image], text=[self.PROMPT], return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs

    def _inner_test_save_load(self, tmpdir):
        super()._inner_test_save_load(tmpdir)
        # Test loading from nested config
        _ = self.RBLN_CLASS.from_pretrained(
            tmpdir,
            export=False,
            rbln_config={"language_model": {"create_runtimes": False}},
            **self.HF_CONFIG_KWARGS,
        )


class TestPegasusModel(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForSeq2SeqLM
    RBLN_CLASS = RBLNPegasusForConditionalGeneration

    # FIXME:: Update to internal once enabled tiny model
    # HF_MODEL_ID = "hf-tiny-model-private/tiny-random-PegasusForConditionalGeneration"
    HF_MODEL_ID = "google/pegasus-xsum"
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
        "decoder_layers": 1,
        "encoder_layers": 1,
    }
    RBLN_CLASS_KWARGS = {"rbln_config": {"enc_max_seq_len": 512, "dec_max_seq_len": 512}}
    PROMPT = "summarize: studies have shown that owning a dog is good for you"
    EXPECTED_OUTPUT = "The The The The The The The The The The The The The The The The The The The"
    TEST_LEVEL = TestLevel.ESSENTIAL

    def get_inputs(self):
        inputs = self.get_tokenizer()(
            self.PROMPT, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        inputs["max_new_tokens"] = 20
        inputs["num_beams"] = 1
        return inputs

    def postprocess(self, inputs, output):
        generated_text = self.get_tokenizer().decode(
            output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return generated_text


class TestLlavaNextForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNLlavaNextForConditionalGeneration
    HF_MODEL_ID = "trl-internal-testing/tiny-LlavaNextForConditionalGeneration"
    PROMPT = "[INST] <image>\nWhat’s shown in this image? [/INST]"
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "language_model": {"use_inputs_embeds": True},
        }
    }
    EXPECTED_OUTPUT = "entricCallbackavidARYails NotesDAPimil coordFeed Boysaml obligation relay迟 войны sexual Definition Eisen patent"
    HF_CONFIG_KWARGS = {"revision": "21948c1af6a0666e341b6403dc1cbbd5c8900e7d"}

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoProcessor.from_pretrained(cls.HF_MODEL_ID, revision=cls.HF_CONFIG_KWARGS["revision"])
        return cls._tokenizer

    # override
    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID, revision=cls.HF_CONFIG_KWARGS["revision"])

        text_config = json.loads(config.text_config.to_json_string())
        text_config["num_hidden_layers"] = 1
        kwargs = {"text_config": text_config}
        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        inputs = tokenizer(images=[image], text=[self.PROMPT], return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs

    def _inner_test_save_load(self, tmpdir):
        super()._inner_test_save_load(tmpdir)
        # Test loading from nested config
        _ = self.RBLN_CLASS.from_pretrained(
            tmpdir,
            export=False,
            rbln_config={"language_model": {"create_runtimes": False}},
            **self.HF_CONFIG_KWARGS,
        )


class TestBlip2ForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNBlip2ForConditionalGeneration
    HF_MODEL_ID = "Salesforce/blip2-opt-2.7b"  # No tiny model yet.
    PROMPT = "Question: Describe this image? Answer:"
    RBLN_CLASS_KWARGS = {"rbln_config": {"language_model": {"use_inputs_embeds": True, "max_seq_len": 1024}}}
    EXPECTED_OUTPUT = "::::::::::::::::::::"
    HF_CONFIG_KWARGS = {}  # Initialize empty to avoid sharing with other classes

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoProcessor.from_pretrained(cls.HF_MODEL_ID)
        return cls._tokenizer

    # override
    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID)

        text_config = json.loads(config.text_config.to_json_string())
        text_config["num_hidden_layers"] = 1
        kwargs = {"text_config": text_config}

        qformer_config = json.loads(config.qformer_config.to_json_string())
        qformer_config["num_hidden_layers"] = 1
        kwargs["qformer_config"] = qformer_config

        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        inputs = tokenizer(images=image, text=self.PROMPT, return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs

    def _inner_test_save_load(self, tmpdir):
        super()._inner_test_save_load(tmpdir)
        # Test loading from nested config
        _ = self.RBLN_CLASS.from_pretrained(
            tmpdir,
            export=False,
            rbln_config={"language_model": {"create_runtimes": False}},
            **self.HF_CONFIG_KWARGS,
        )


class TestIdefics3ForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNIdefics3ForConditionalGeneration
    HF_MODEL_ID = "hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration"
    PROMPT = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    RBLN_CLASS_KWARGS = {"rbln_config": {"text_model": {"use_inputs_embeds": True, "attn_impl": "flash_attn"}}}
    HF_CONFIG_KWARGS = {}  # Initialize empty to avoid sharing with other classes

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoProcessor.from_pretrained(cls.HF_MODEL_ID)
        return cls._tokenizer

    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID)
        text_config = json.loads(config.text_config.to_json_string())
        text_config["num_hidden_layers"] = 1
        kwargs = {"text_config": text_config}
        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        text = tokenizer.apply_chat_template(self.PROMPT, add_generation_prompt=True)
        inputs = tokenizer(images=[image], text=[text], return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs


class TestQwen2_5_VLForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForVision2Seq
    RBLN_CLASS = RBLNQwen2_5_VLForConditionalGeneration
    HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # No tiny model yet.
    PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "visual": {"max_seq_lens": 512},
            "tensor_parallel_size": 1,
            "kvcache_partition_len": 16_384,
            "max_seq_len": 32_768,
        }
    }
    EXPECTED_OUTPUT = "讣讣讣讣讣讣讣讣讣讣讣讣讣讣讣讣讣讣讣讣"
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 1,
    }

    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID)
        vision_config = json.loads(config.vision_config.to_json_string())
        vision_config["depth"] = 8
        vision_config["fullatt_block_indexes"] = [7]
        kwargs = {"vision_config": vision_config}
        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoProcessor.from_pretrained(cls.HF_MODEL_ID, max_pixels=64 * 14 * 14)
        return cls._tokenizer

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        inputs = tokenizer(images=[image], text=[self.PROMPT], return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs


class TestGemma3ForConditionalGeneration(LLMTest.TestLLM):
    RBLN_AUTO_CLASS = RBLNAutoModelForImageTextToText
    RBLN_CLASS = RBLNGemma3ForConditionalGeneration
    HF_MODEL_ID = "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"  # No tiny model yet.
    PROMPT = "<bos><start_of_turn>user\n<start_of_image>Describe the image.<end_of_turn>\n<start_of_turn>model\n'"
    RBLN_CLASS_KWARGS = {"rbln_config": {"language_model": {"use_inputs_embeds": True, "kvcache_partition_len": 4096}}}
    EXPECTED_OUTPUT = " அனுமதி Bryson Earlyheiserheiserheiserheiserheiserheiserheiserheiserheiserheiserheiserheiserheiserheiserheiserिल्म हस्ता"
    HF_CONFIG_KWARGS = {
        "revision": "e1f4b0516ec80f86ed75c8cb1d45ede72526ad24",
    }
    TEST_LEVEL = TestLevel.FULL

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoProcessor.from_pretrained(cls.HF_MODEL_ID, revision=cls.HF_CONFIG_KWARGS["revision"])
        return cls._tokenizer

    # override
    @classmethod
    def setUpClass(cls):
        config = AutoConfig.from_pretrained(cls.HF_MODEL_ID, revision=cls.HF_CONFIG_KWARGS["revision"])
        text_config = json.loads(config.text_config.to_json_string())
        text_config["num_hidden_layers"] = 2
        text_config["sliding_window_pattern"] = 2
        vision_config = json.loads(config.vision_config.to_json_string())
        vision_config["num_hidden_layers"] = 1
        kwargs = {"text_config": text_config, "vision_config": vision_config}
        cls.HF_CONFIG_KWARGS.update(kwargs)
        return super().setUpClass()

    def get_inputs(self):
        tokenizer = self.get_tokenizer()
        img_path = f"{os.path.dirname(__file__)}/../assets/rbln_logo.png"
        image = Image.open(img_path)
        image = image.convert("RGB")
        inputs = tokenizer(images=[image], text=[self.PROMPT], return_tensors="pt", padding=True)
        inputs["max_new_tokens"] = 20
        inputs["do_sample"] = False
        return inputs


class TestGemma3ForCausalLM(LLMTest.TestLLM):
    RBLN_CLASS = RBLNGemma3ForCausalLM
    HF_MODEL_ID = "google/gemma-3-1b-it"
    EXPECTED_OUTPUT = "1st L L L L L L L L L L L L L L L L L L"
    HF_CONFIG_KWARGS = {
        "num_hidden_layers": 2,
        "sliding_window_pattern": 2,
        "max_position_embeddings": 1024,
        "trust_remote_code": True,
    }


class TestLlamaForCausalLM_fp8(LLMTest.TestLLM):
    RBLN_CLASS = RBLNLlamaForCausalLM
    HF_MODEL_ID = "RedHatAI/Meta-Llama-3-8B-Instruct-FP8-KV"  # No tiny model yet.
    HF_CONFIG_KWARGS = {"num_hidden_layers": 1}
    RBLN_CLASS_KWARGS = {
        "rbln_config": {
            "quantization": {"weights": "fp8", "kv_caches": "fp8"},
            "create_runtimes": False,
            "npu": "RBLN-CR03",
            "attn_impl": "flash_attn",
            "kvcache_partition_len": 4096,
            "max_seq_len": 8192,
            "tensor_parallel_size": 1,
        },
    }
    EXPECTED_OUTPUT = None  # Cannot generate output with fp8 quantization in ATOM™
    TEST_LEVEL = TestLevel.DISABLED


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
