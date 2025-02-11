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

import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import HfApi, HfFolder, hf_hub_download


class PushToHubMixin:
    def push_to_hub(
        self,
        save_directory: str,
        repository_id: str,
        private: Optional[bool] = None,
        use_auth_token: Union[bool, str] = True,
    ) -> str:
        huggingface_token = _get_huggingface_token(use_auth_token)
        api = HfApi()

        api.create_repo(
            token=huggingface_token,
            repo_id=repository_id,
            exist_ok=True,
            private=private,
        )
        for path, subdirs, files in os.walk(save_directory):
            for name in files:
                local_file_path = os.path.join(path, name)
                _, hub_file_path = os.path.split(local_file_path)
                # FIXME: when huggingface_hub fixes the return of upload_file
                try:
                    api.upload_file(
                        token=huggingface_token,
                        repo_id=f"{repository_id}",
                        path_or_fileobj=os.path.join(os.getcwd(), local_file_path),
                        path_in_repo=hub_file_path,
                    )
                except KeyError:
                    pass
                except NameError:
                    pass


def pull_compiled_model_from_hub(
    model_id: Union[str, Path],
    subfolder: str,
    use_auth_token: Optional[Union[bool, str]],
    revision: Optional[str],
    cache_dir: Optional[str],
    force_download: bool,
    local_files_only: bool,
) -> Path:
    """Pull model files from the Hugging Face Hub."""
    huggingface_token = _get_huggingface_token(use_auth_token)
    repo_files = list(
        map(
            Path,
            HfApi().list_repo_files(model_id, revision=revision, token=huggingface_token),
        )
    )

    pattern_rbln = "*.rbln" if subfolder == "" else f"{subfolder}/*.rbln"
    rbln_files = [p for p in repo_files if p.match(pattern_rbln)]

    pattern_config = "rbln_config.json" if subfolder == "" else f"{subfolder}/rbln_config.json"
    rbln_config_filenames = [p for p in repo_files if p.match(pattern_config)]

    validate_files(rbln_files, rbln_config_filenames, f"repository {model_id}")

    filenames = [str(path) for path in repo_files]

    for filename in filenames:
        rbln_config_cache_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            subfolder=subfolder,
            use_auth_token=use_auth_token,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
        )

    return Path(rbln_config_cache_path).parent


def validate_files(
    files: List[Path],
    config_files: List[Path],
    location: str,
):
    """Validate the presence and count of required files."""
    if len(files) == 0:
        raise FileNotFoundError(f"Could not find any rbln model file in {location}")

    if len(config_files) == 0:
        raise FileNotFoundError(f"Could not find `rbln_config.json` file in {location}")

    if len(config_files) > 1:
        raise FileExistsError(f"Multiple rbln_config.json files found in {location}. This is not expected.")


def _get_huggingface_token(use_auth_token: Union[bool, str]) -> str:
    if isinstance(use_auth_token, str):
        return use_auth_token
    elif use_auth_token:
        return HfFolder.get_token()
    else:
        raise ValueError("`use_auth_token` must be provided to interact with the Hugging Face Hub.")
