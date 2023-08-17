#!/bin/bash

# There is no stable way to handle dynamic versions on lock files.
# See https://github.com/astral-sh/uv/issues/7533 for the detailed discussion.
#
# Until the above issue is resolved, we use workaround suggested in the below link
# to avoid locking with a dynamic (hatch-vsc) version.
# https://github.com/astral-sh/uv/issues/7533#issuecomment-2486235749

export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
uv lock --refresh
