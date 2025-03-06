import filecmp
import inspect
import os
import random
import shutil
import tempfile
import unittest
from enum import Enum
from pathlib import Path

import transformers
from diffusers import DiffusionPipeline
from transformers import AutoConfig, CLIPConfig


class TestLevel(Enum):
    ESSENTIAL = 1
    DEFAULT = 2
    FULL = 3
    UNKNOWN = -1


def require_hf_user_id(test_case):
    """
    Decorator marking a test that requires huggingface hub user id.
    """
    user_id = os.environ.get("HF_USER_ID", None)
    if user_id is None:
        return unittest.skip("test requires hf token as `HF_USER_ID` environment variable")(test_case)
    else:
        return test_case


def require_hf_token(test_case):
    """
    Decorator marking a test that requires huggingface hub token.
    """
    use_auth_token = os.environ.get("HF_AUTH_TOKEN", None)
    if use_auth_token is None:
        return unittest.skip("test requires hf token as `HF_AUTH_TOKEN` environment variable")(test_case)
    else:
        return test_case


class BaseTest:
    """
    Base Class for other models.

    You should specify class attributes : RBLN_CLASS, HF_MODEL_ID
    """

    class TestModel(unittest.TestCase):
        RBLN_AUTO_CLASS = None
        RBLN_CLASS = None
        HF_MODEL_ID = None
        RBLN_CLASS_KWARGS = {}
        GENERATION_KWARGS = {}
        HF_CONFIG_KWARGS = {}
        EXPECTED_OUTPUT = None
        TEST_LEVEL = TestLevel.DEFAULT
        DEVICE = -1  # -1 indicates dummy device

        @classmethod
        def setUpClass(cls):
            env_coverage = os.environ.get("OPTIMUM_RBLN_TEST_LEVEL", "default")
            env_coverage = TestLevel[env_coverage.upper()]
            if env_coverage.value < cls.TEST_LEVEL.value:
                raise unittest.SkipTest(f"Skipped test : Test Coverage {env_coverage.name} < {cls.TEST_LEVEL.name}")

            if os.path.exists(cls.get_rbln_local_dir()):
                shutil.rmtree(cls.get_rbln_local_dir())

            cls.model = cls.RBLN_CLASS.from_pretrained(
                cls.HF_MODEL_ID,
                export=True,
                model_save_dir=cls.get_rbln_local_dir(),
                rbln_device=cls.DEVICE,
                **cls.RBLN_CLASS_KWARGS,
                **cls.HF_CONFIG_KWARGS,
            )

        @classmethod
        def get_rbln_local_dir(cls):
            return os.path.basename(cls.HF_MODEL_ID) + "-local"

        @classmethod
        def get_hf_auto_class(cls):
            if cls.RBLN_AUTO_CLASS is not None:
                return getattr(transformers, cls.RBLN_AUTO_CLASS.__name__[4:])
            return None

        @classmethod
        def get_hf_class(cls):
            return getattr(transformers, cls.RBLN_CLASS.__name__[4:])

        @classmethod
        def get_hf_remote_dir(cls):
            return "rbln-" + os.path.basename(cls.HF_MODEL_ID)

        def is_diffuser(self):
            # Note that This is only True when it is a pipeline, not model (i.e. AutoEncoderKL)
            return isinstance(self.model, DiffusionPipeline)

        @classmethod
        def tearDownClass(cls):
            if os.path.exists(cls.get_rbln_local_dir()):
                shutil.rmtree(cls.get_rbln_local_dir())

        def test_model_save_dir(self):
            self.assertTrue(os.path.exists(self.get_rbln_local_dir()), "model_save_dir does not work.")

        @require_hf_token
        @require_hf_user_id
        def test_push_to_hub(self):
            """
            "HF_AUTH_TOKEN" should be set to execute this.
            """
            with tempfile.TemporaryDirectory() as tmpdirname:
                # create remote hash to check if file was updated.

                remote_hash = random.getrandbits(128)
                HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", None)
                HF_USER_ID = os.environ.get("HF_USER_ID", None)

                self.assertFalse(HF_AUTH_TOKEN is None)
                self.assertFalse(HF_USER_ID is None)

                if self.is_diffuser():
                    TOKEN_KEY = "token"
                    REPO_KEY = "repo_id"
                    self.model.text_encoder.config.from_local = remote_hash
                else:
                    TOKEN_KEY = "use_auth_token"
                    REPO_KEY = "repository_id"
                    self.model.config.from_local = remote_hash

                self.model.save_pretrained(
                    tmpdirname,
                    push_to_hub=True,
                    private=True,
                    **{
                        TOKEN_KEY: HF_AUTH_TOKEN,
                        REPO_KEY: f"{HF_USER_ID}/{self.get_hf_remote_dir()}",
                    },
                )

                # If our tests were moved to a public rather than a private repository,
                # this logic could be as simple as downloading the config file directly
                # and comparing it.
                if self.is_diffuser():
                    cfg = CLIPConfig.from_pretrained(
                        f"{HF_USER_ID}/{self.get_hf_remote_dir()}",
                        subfolder="text_encoder",
                        private=True,
                        **{TOKEN_KEY: HF_AUTH_TOKEN},
                    )
                else:
                    cfg = AutoConfig.from_pretrained(
                        f"{HF_USER_ID}/{self.get_hf_remote_dir()}",
                        private=True,
                        **{TOKEN_KEY: HF_AUTH_TOKEN},
                    )

                self.assertEqual(remote_hash, cfg.from_local)

        def get_inputs(self):
            return self.GENERATION_KWARGS

        def postprocess(self, inputs, output):
            return output

        def test_generate(self):
            inputs = self.get_inputs()
            if self.is_diffuser():
                output = self.model(**inputs)[0]
            else:
                if self.model.can_generate():
                    output = self.model.generate(**inputs)
                else:
                    # encoder-only, resnet, etc..
                    output = self.model(**inputs)

            output = self.postprocess(inputs, output)
            if self.EXPECTED_OUTPUT:
                self.assertEqual(output, self.EXPECTED_OUTPUT)

        def test_save_load(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.subTest():
                    self.model.save_pretrained(tmpdir)
                    config_path = os.path.join(tmpdir, self.RBLN_CLASS.config_name)
                    self.assertTrue(os.path.exists(config_path), "save_pretrained does not work.")

                with self.subTest():
                    # Test load
                    _ = self.RBLN_CLASS.from_pretrained(
                        tmpdir,
                        export=False,
                        rbln_create_runtimes=False,
                        **self.HF_CONFIG_KWARGS,
                    )

                with self.subTest():
                    # Test saving from exported pipe
                    self.model.save_pretrained(tmpdir)
                    _ = self.RBLN_CLASS.from_pretrained(
                        tmpdir,
                        export=False,
                        rbln_create_runtimes=False,
                        **self.HF_CONFIG_KWARGS,
                    )

        def test_model_save_dir_load(self):
            # Test model_save_dir
            _ = self.RBLN_CLASS.from_pretrained(
                self.get_rbln_local_dir(),
                export=False,
                rbln_create_runtimes=False,
                **self.HF_CONFIG_KWARGS,
            )

        def test_automap(self):
            if self.RBLN_AUTO_CLASS is None:
                self.skipTest("Skipping test because RBLN_AUTO_CLASS is None")
            assert self.RBLN_CLASS == self.RBLN_AUTO_CLASS.get_rbln_cls(
                self.HF_MODEL_ID,
                **self.RBLN_CLASS_KWARGS,
                **self.HF_CONFIG_KWARGS,
            )

        # check if this use a pipeline
        def test_infer_framework(self):
            class_hierarchy = inspect.getmro(self.model.__class__)

            is_valid_framework = (
                any(base_class.__name__ == "PreTrainedModel" for base_class in class_hierarchy)
                if any("transformers" in str(base_class) for base_class in class_hierarchy)
                else True
            )

            assert is_valid_framework, "Model does not inherit from PreTrainedModel."

        @require_hf_token
        @require_hf_user_id
        def test_pull_compiled_model_from_hub(self):
            HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", None)
            HF_USER_ID = os.environ.get("HF_USER_ID", None)

            pull_model_dir = self.RBLN_CLASS._load_compiled_model_dir(
                f"{HF_USER_ID}/{self.get_hf_remote_dir()}",
                HF_AUTH_TOKEN,
            )

            path = Path(self.get_rbln_local_dir())
            num_files = sum(1 for _ in path.rglob("*") if _.is_file())

            assert len(filecmp.dircmp(pull_model_dir, self.get_rbln_local_dir()).common) == num_files


class DisallowedTestBase:
    class DisallowedTest(unittest.TestCase):
        RBLN_CLASS = None
        HF_MODEL_ID = None
        RBLN_CLASS_KWARGS = {}
        GENERATION_KWARGS = {}
        HF_CONFIG_KWARGS = {}
        TEST_LEVEL = TestLevel.DEFAULT

        @classmethod
        def setUpClass(cls):
            env_coverage = os.environ.get("OPTIMUM_RBLN_TEST_LEVEL", "default")
            env_coverage = TestLevel[env_coverage.upper()]
            if env_coverage.value < cls.TEST_LEVEL.value:
                raise unittest.SkipTest(f"Skipped test : Test Coverage {env_coverage.name} < {cls.TEST_LEVEL.name}")

        def test_load(self):
            try:
                _ = self.RBLN_CLASS.from_pretrained(
                    self.HF_MODEL_ID,
                    export=True,
                    model_save_dir=self.get_rbln_local_dir(),
                    **self.RBLN_CLASS_KWARGS,
                    **self.HF_CONFIG_KWARGS,
                )

                self.assertTrue(False, "This should be disallowed.")

            except ValueError:
                pass

            finally:
                if os.path.exists(self.get_rbln_local_dir()):
                    shutil.rmtree(self.get_rbln_local_dir())

        @classmethod
        def get_rbln_local_dir(cls):
            return os.path.basename(cls.HF_MODEL_ID) + "-local"
