#!/usr/bin/env python3
# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import shutil
import sys
from pathlib import Path

from .utils.model_utils import get_rbln_model_cls
from .utils.runtime_utils import ContextRblnConfig


def set_nested_dict(dictionary, key_path, value):
    """
    Set a value in a nested dictionary using dot notation.

    Args:
        dictionary: The dictionary to modify
        key_path: Dot-separated key path (e.g., "unet.batch_size")
        value: The value to set
    """
    keys = key_path.split(".")
    current = dictionary

    # Navigate to the parent of the final key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the final value
    current[keys[-1]] = value


def parse_value(value_str):
    """
    Parse a string value to appropriate Python type.

    Args:
        value_str: String value to parse

    Returns:
        Parsed value (bool, int, float, or str)
    """
    if value_str.lower() in ["true", "false"]:
        return value_str.lower() == "true"
    elif value_str.isdigit():
        return int(value_str)
    else:
        try:
            return float(value_str)
        except ValueError:
            return value_str


def main():
    """
    Main CLI function for optimum-rbln model compilation.
    """
    parser = argparse.ArgumentParser(
        description="Compile models for RBLN devices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile Llama model with specific class
  optimum-rbln-cli ./compiled_model_dir --class RBLNLlamaForCausalLM --model-id meta-llama/Llama-2-7b-chat-hf --batch-size 2 --tensor-parallel-size 4

  # Compile BERT model for sequence classification
  optimum-rbln-cli ./bert_compiled --class RBLNBertForSequenceClassification --model-id bert-base-uncased --batch-size 8 --max-seq-len 512

  # Use auto model class
  optimum-rbln-cli ./auto_compiled --class RBLNAutoModelForCausalLM --model-id gpt2 --batch-size 1

  # Use nested rbln_config arguments
  optimum-rbln-cli ./stable_diffusion_compiled --class RBLNStableDiffusionPipeline --model-id runwayml/stable-diffusion-v1-5 --unet.batch_size 2 --vae.batch_size 1
        """,
    )

    # Required positional argument
    parser.add_argument("output_dir", type=str, help="Directory where the compiled model will be saved")

    # Required arguments
    parser.add_argument(
        "--class",
        dest="model_class",
        type=str,
        required=True,
        help="RBLN model class to use for compilation (e.g., RBLNLlamaForCausalLM, RBLNAutoModelForCausalLM)",
    )

    parser.add_argument(
        "--model-id",
        dest="model_id",
        type=str,
        required=True,
        help="Model ID from HuggingFace Hub or local directory path",
    )

    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists",
    )
    # All other arguments will be parsed dynamically and passed to from_pretrained

    # Parse known args to allow for additional rbln_* arguments
    args, unknown_args = parser.parse_known_args()

    try:
        # Get the model class using the utility function
        model_class = get_rbln_model_cls(args.model_class)

        # Create output directory
        output_path = Path(args.output_dir)
        if output_path.exists():
            if args.overwrite:
                shutil.rmtree(output_path)
            else:
                raise FileExistsError(f"Output directory {output_path} already exists")

        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare rbln_config by parsing all unknown arguments
        rbln_config = {}

        # Parse all unknown arguments
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]
            if arg.startswith("--"):
                arg_name = arg[2:].replace("-", "_")
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                    # Has a value
                    arg_value = unknown_args[i + 1]
                    parsed_value = parse_value(arg_value)

                    # Check if this is a nested config argument (contains dots)
                    if "." in arg_name:
                        set_nested_dict(rbln_config, arg_name, parsed_value)
                    else:
                        rbln_config[arg_name] = parsed_value
                    i += 2
                else:
                    # Boolean flag
                    if "." in arg_name:
                        set_nested_dict(rbln_config, arg_name, True)
                    else:
                        rbln_config[arg_name] = True
                    i += 1
            else:
                i += 1

        # Set create_runtimes to False by default for CLI compilation if not specified
        create_runtimes = rbln_config.pop("create_runtimes", False)

        print(f"Compiling model '{args.model_id}' using {args.model_class}...")
        print(f"Output directory: {output_path.absolute()}")
        print(f"RBLN Config: {rbln_config}")

        with ContextRblnConfig(create_runtimes=create_runtimes):
            # Compile the model - pass all arguments as rbln_config
            compiled_model = model_class.from_pretrained(
                args.model_id, export=True, model_save_dir=str(output_path), rbln_config=rbln_config
            )

        print("✅ Model compilation completed successfully!")
        print(f"Compiled model saved to: {output_path.absolute()}")

        # Print model info
        if hasattr(compiled_model, "rbln_config"):
            print(f"Batch size: {compiled_model.rbln_config.batch_size}")
            if hasattr(compiled_model.rbln_config, "tensor_parallel_size"):
                print(f"Tensor parallel size: {compiled_model.rbln_config.tensor_parallel_size}")
            if hasattr(compiled_model.rbln_config, "max_seq_len"):
                print(f"Max sequence length: {compiled_model.rbln_config.max_seq_len}")

    except Exception as e:
        print(f"❌ Error during model compilation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
