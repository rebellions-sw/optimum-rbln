# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this software are licensed under the Apache License,
# Version 2.0. See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.

# All other portions of this software, including proprietary code,
# are the intellectual property of Rebellions Inc. and may not be
# copied, modified, or distributed without prior written permission
# from Rebellions Inc.

import importlib.metadata
import importlib.util
import warnings
from dataclasses import dataclass

from packaging.version import Version


@dataclass
class VersionCompat:
    package_name: str
    min_version: str
    max_version: str


RBLN_VERSION_COMPATS = {
    "0.2.0": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.7.1",
            max_version="0.7.2",
        ),
    ],
    "0.1.15": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.6.2",
            max_version="0.6.3",
        ),
    ],
    "0.1.14": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.6.2",
            max_version="0.6.3",
        ),
    ],
    "0.1.13": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.6.0",
            max_version="0.6.2",
        ),
    ],
    "0.1.12": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.5.12",
            max_version="0.5.13",
        ),
    ],
    "0.1.11": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.5.10",
            max_version="0.5.11",
        ),
    ],
    "0.1.10": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.5.10",
            max_version="0.5.11",
        ),
    ],
    "0.1.9": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.5.9",
            max_version="0.5.10",
        ),
    ],
    "0.1.8": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.5.8",
            max_version="0.5.9",
        ),
    ],
    "0.1.7": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.5.7",
            max_version="0.5.8",
        ),
    ],
    "0.1.4": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.5.2",
            max_version="0.5.3",
        ),
    ],
    "0.1.0": [
        VersionCompat(
            package_name="rebel-compiler",
            min_version="0.5.0",
            max_version="0.5.1",
        ),
    ],
    "0.0.0": [],
}


def is_rbln_available() -> bool:
    return importlib.util.find_spec("rebel-compiler") is not None


def check_version_compats() -> None:
    warnings.filterwarnings(action="always", category=ImportWarning, module="optimum.rbln")
    my_version = importlib.metadata.version("optimum-rbln")
    target_version = list(filter(lambda v: Version(my_version) > Version(v), RBLN_VERSION_COMPATS.keys()))[0]
    for compat in RBLN_VERSION_COMPATS[target_version]:
        try:
            dep_version = importlib.metadata.version(compat.package_name)
        except importlib.metadata.PackageNotFoundError:
            warnings.warn(f"optimum-rbln requires {compat.package_name} to be installed.", ImportWarning)
            continue

        if not Version(compat.min_version) <= Version(dep_version) < Version(compat.max_version):
            warnings.warn(
                f"optimum-rbln v{my_version} is compatible to {compat.package_name} v{compat.min_version} to v{compat.max_version}. (you are currently using v{dep_version})\n"
                "Please refer to our SDK release notes at https://docs.rbln.ai/about_atom/release_note.html",
                ImportWarning,
            )
