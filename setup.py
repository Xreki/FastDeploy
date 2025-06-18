"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

long_description = "FastDeploy: Large Language Model Serving.\n\n"
long_description += "GitHub: https://github.com/PaddlePaddle/FastDeploy\n"
long_description += "Email: dltp@baidu.com"

# Platform to CMake mapping
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):
    """A setuptools Extension for CMake-based builds."""

    def __init__(self, name: str, sourcedir: str = "") -> None:
        """
        Initialize CMake extension.

        Args:
            name (str): Name of the extension.
            sourcedir (str): Source directory path.
        """
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    """Custom build_ext command using CMake."""

    def build_extension(self, ext: CMakeExtension) -> None:
        """
        Build the CMake extension.

        Args:
            ext (CMakeExtension): The extension to build.
        """
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        cfg = "Debug" if int(os.environ.get("DEBUG", 0)) else "Release"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DVERSION_INFO={self.distribution.get_version()}"
        ]
        build_args = []

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        if self.compiler.compiler_type != "msvc":
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja
                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}"
                    ]
                except ImportError:
                    pass
        else:
            if "NMake" not in cmake_generator and "Ninja" not in cmake_generator:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]
            if "NMake" not in cmake_generator and "Ninja" not in cmake_generator:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))
                ]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ and hasattr(
                self, "parallel") and self.parallel:
            build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(["cmake", ext.sourcedir, *cmake_args],
                       cwd=build_temp,
                       check=True)
        subprocess.run(["cmake", "--build", ".", *build_args],
                       cwd=build_temp,
                       check=True)


def load_requirements():
    """加载requirements.txt中的依赖"""
    requirements_path = os.path.join(os.path.dirname(__file__),
                                     'requirements.txt')
    with open(requirements_path, 'r') as f:
        return [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]


setuptools.setup(
    name="fastdeploy",
    version="2.0.0-alpha",
    author="PaddlePaddle",
    author_email="dltp@baidu.com",
    description="FastDeploy: Large Language Model Serving.",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/PaddlePaddle/FastDeploy",
    packages=setuptools.find_packages(),
    package_dir={"fastdeploy": "fastdeploy/"},
    package_data={
        "fastdeploy": [
            "model_executor/ops/gpu/*",
            "model_executor/ops/gpu/deep_gemm/include/**/*",
            "model_executor/ops/cpu/*", "model_executor/ops/xpu/*",
            "model_executor/ops/npu/*", "model_executor/ops/base/*",
            "model_executor/models/*", "model_executor/layers/*",
            "input/mm_processor/utils/*"
        ]
    },
    install_requires=load_requirements(),
    ext_modules=[
        CMakeExtension(
            "rdma_comm",
            sourcedir=
            "fastdeploy/cache_manager/transfer_factory/kvcache_transfer")
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    python_requires=">=3.7",
    extras_require={"test": ["pytest>=6.0"]},
)
