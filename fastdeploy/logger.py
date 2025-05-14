# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""logger"""

import os
import logging

# ANSI escape sequences for colors
COLORS = {
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "DEBUG": "\033[35m",  # Purple
    "ERROR": "\033[31m",  # Red
    "RESET": "\033[0m",  # Reset to default color
}


class ColoredFormatter(logging.Formatter):
    """ColoredFormatter"""

    def format(self, record):
        # Apply color only to the level name
        levelname_colored = f"{COLORS.get(record.levelname, COLORS['RESET'])}{record.levelname}{COLORS['RESET']}"
        # Replace the original levelname with the colored one
        record.levelname = levelname_colored
        # Use the super class's format method
        return super().format(record)


def setup_logger():
    """
    Setup logger for efficientllm runtime.
    """
    # Create a logger
    logger = logging.getLogger("efficientllm_runtime_log")
    log_level = int(os.getenv("ELLM_LOG_LEVEL", 1))
    if log_level == 0:
        logger.setLevel(logging.DEBUG)
    elif log_level == 1:
        logger.setLevel(logging.INFO)
    elif log_level == 2:
        logger.setLevel(logging.WARN)
    else:
        logger.setLevel(logging.ERROR)
    logger.propagate = False
    if not logger.handlers:
        ch = logging.StreamHandler()
        # Define a format that matches the requested style
        formatter = ColoredFormatter(
            "%(levelname)-8s %(asctime)s %(process)d %(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


logger = setup_logger()
