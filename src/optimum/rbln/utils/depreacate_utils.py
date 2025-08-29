from typing import Optional

import rebel

from .logging import get_logger


logger = get_logger(__name__)


def warn_deprecated_npu(npu: Optional[str] = None):
    npu = npu or rebel.get_npu_name()
    if npu == "RBLN-CA02":
        logger.warning_once(
            "Support for the RBLN-CA02 device is provided only up to optimum-rbln v0.8.0 and has reached end of life.",
        )
