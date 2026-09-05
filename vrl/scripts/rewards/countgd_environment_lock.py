"""Qualified distribution lock for the CountGD HTTP reward runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

ENVIRONMENT_LOCK_SCHEMA = "vrl.countgd-python-lock/v3"
QUALIFIED_PYTHON_VERSION = "3.12.2"
QUALIFIED_PLATFORM = "Linux"
QUALIFIED_MACHINE = "x86_64"
QUALIFIED_PIP_VERSION = "24.0"


class ArtifactSource(StrEnum):
    """Authoritative index that published an artifact."""

    PYPI = "pypi"
    PYTORCH = "pytorch"


class ArtifactKind(StrEnum):
    """Packaging form supplied by the authoritative index."""

    WHEEL = "wheel"
    SDIST = "sdist"


@dataclass(frozen=True, slots=True)
class LockedDistribution:
    """One exact package artifact accepted by the qualified target."""

    name: str
    version: str
    filename: str
    url: str
    sha256: str
    source: ArtifactSource = ArtifactSource.PYPI
    artifact_kind: ArtifactKind = ArtifactKind.WHEEL

    @property
    def requirement(self) -> str:
        return f"{self.name}=={self.version}"


# This deliberately isolated table is the complete active model plus standalone
# HTTP-service closure for CPython 3.12.2 on Linux x86_64. CUDA artifacts are
# present because Torch's Linux marker is active on the only accepted target.
# pip is a separate venv bootstrap invariant. antlr 4.9.3 is the sole sdist
# because OmegaConf 2.3.0 requires it and PyPI publishes no wheel for that
# version; it is built offline with the locked setuptools already installed.
ENVIRONMENT_LOCK = (
    LockedDistribution(
        name="addict",
        version="2.4.0",
        filename="addict-2.4.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/6a/00/b08f23b7d7e1e14ce01419a467b583edbb93c6cdb8654e54a9cc579cd61f/addict-2.4.0-py3-none-any.whl",
        sha256="249bb56bbfd3cdc2a004ea0ff4c2b6ddc84d53bc2194761636eb314d5cfa5dfc",
    ),
    LockedDistribution(
        name="aiohappyeyeballs",
        version="2.6.1",
        filename="aiohappyeyeballs-2.6.1-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/0f/15/5bf3b99495fb160b63f95972b81750f18f7f4e02ad051373b669d17d44f2/aiohappyeyeballs-2.6.1-py3-none-any.whl",
        sha256="f349ba8f4b75cb25c99c5c2d84e997e485204d2902a9597802b0371f09331fb8",
    ),
    LockedDistribution(
        name="aiohttp",
        version="3.14.3",
        filename="aiohttp-3.14.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        url="https://files.pythonhosted.org/packages/52/b7/7cd31f29d6055bd711ae6e669367fba6f5ae9de463910a793e30556a8db7/aiohttp-3.14.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        sha256="543906c127fb1d929b95076db19b83fa2d46751006ff1e23b093aa5ac4d8db42",
    ),
    LockedDistribution(
        name="aiosignal",
        version="1.4.0",
        filename="aiosignal-1.4.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/fb/76/641ae371508676492379f16e2fa48f4e2c11741bd63c48be4b12a6b09cba/aiosignal-1.4.0-py3-none-any.whl",
        sha256="053243f8b92b990551949e63930a839ff0cf0b0ebbe0597b0f3fb19e1a0fe82e",
    ),
    LockedDistribution(
        name="antlr4-python3-runtime",
        version="4.9.3",
        filename="antlr4-python3-runtime-4.9.3.tar.gz",
        url="https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/antlr4-python3-runtime-4.9.3.tar.gz",
        sha256="f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b",
        artifact_kind=ArtifactKind.SDIST,
    ),
    LockedDistribution(
        name="attrs",
        version="25.1.0",
        filename="attrs-25.1.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/fc/30/d4986a882011f9df997a55e6becd864812ccfcd821d64aac8570ee39f719/attrs-25.1.0-py3-none-any.whl",
        sha256="c75a69e28a550a7e93789579c22aa26b0f5b83b75dc4e08fe092980051e1090a",
    ),
    LockedDistribution(
        name="certifi",
        version="2026.1.4",
        filename="certifi-2026.1.4-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/e6/ad/3cc14f097111b4de0040c83a525973216457bbeeb63739ef1ed275c1c021/certifi-2026.1.4-py3-none-any.whl",
        sha256="9943707519e4add1115f44c2bc244f782c0249876bf51b6599fee1ffbedd685c",
    ),
    LockedDistribution(
        name="charset-normalizer",
        version="3.3.2",
        filename="charset_normalizer-3.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        url="https://files.pythonhosted.org/packages/ee/fb/14d30eb4956408ee3ae09ad34299131fb383c47df355ddb428a7331cfa1e/charset_normalizer-3.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        sha256="90d558489962fd4918143277a773316e56c72da56ec7aa3dc3dbbe20fdfed15b",
    ),
    LockedDistribution(
        name="colorlog",
        version="6.12.0",
        filename="colorlog-6.12.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/d4/19/0b6647bf5e331521e55d2b63bfbdc210bd9cd605189273f03614a05f702d/colorlog-6.12.0-py3-none-any.whl",
        sha256="30d392604e9110045a2c2aeefc27d7a017abbab63f3a8aee594eac0801df784e",
    ),
    LockedDistribution(
        name="contourpy",
        version="1.3.2",
        filename="contourpy-1.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        url="https://files.pythonhosted.org/packages/a8/32/b8a1c8965e4f72482ff2d1ac2cd670ce0b542f203c8e1d34e7c3e6925da7/contourpy-1.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        sha256="f26b383144cf2d2c29f01a1e8170f50dacf0eac02d64139dcd709a8ac4eb3cfe",
    ),
    LockedDistribution(
        name="cuda-bindings",
        version="12.9.4",
        filename="cuda_bindings-12.9.4-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        url="https://files.pythonhosted.org/packages/a9/c1/dabe88f52c3e3760d861401bb994df08f672ec893b8f7592dc91626adcf3/cuda_bindings-12.9.4-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        sha256="fda147a344e8eaeca0c6ff113d2851ffca8f7dfc0a6c932374ee5c47caa649c8",
    ),
    LockedDistribution(
        name="cuda-pathfinder",
        version="1.3.3",
        filename="cuda_pathfinder-1.3.3-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/0b/02/4dbe7568a42e46582248942f54dc64ad094769532adbe21e525e4edf7bc4/cuda_pathfinder-1.3.3-py3-none-any.whl",
        sha256="9984b664e404f7c134954a771be8775dfd6180ea1e1aef4a5a37d4be05d9bbb1",
    ),
    LockedDistribution(
        name="cuda-toolkit",
        version="12.8.1",
        filename="cuda_toolkit-12.8.1-py2.py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/d4/c8/7dce3a0b15b42a3b58e7d96eb22a687d3bf2c44e01d149a6874629cd9938/cuda_toolkit-12.8.1-py2.py3-none-any.whl",
        sha256="adc7906af4ecbf9a352f9dca5734eceb21daec281ccfcf5675e1d2f724fc2cba",
    ),
    LockedDistribution(
        name="cycler",
        version="0.12.1",
        filename="cycler-0.12.1-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/e7/05/c19819d5e3d95294a6f5947fb9b9629efb316b96de511b418c53d245aae6/cycler-0.12.1-py3-none-any.whl",
        sha256="85cef7cff222d8644161529808465972e51340599459b8ac3ccbac5a854e0d30",
    ),
    LockedDistribution(
        name="filelock",
        version="3.29.4",
        filename="filelock-3.29.4-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/13/37/a065dc3bd6e49423a6532c642ca7378d3f467b1ef44c2800c937af7f9739/filelock-3.29.4-py3-none-any.whl",
        sha256="dac1648087d5115554850d113e7dd8c83ab2d38e3435dde2d4f163847e57b767",
    ),
    LockedDistribution(
        name="fonttools",
        version="4.61.1",
        filename="fonttools-4.61.1-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl",
        url="https://files.pythonhosted.org/packages/b7/37/82dbef0f6342eb01f54bca073ac1498433d6ce71e50c3c3282b655733b31/fonttools-4.61.1-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl",
        sha256="10d88e55330e092940584774ee5e8a6971b01fc2f4d3466a1d6c158230880796",
    ),
    LockedDistribution(
        name="frozenlist",
        version="1.8.0",
        filename="frozenlist-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl",
        url="https://files.pythonhosted.org/packages/6a/bd/d91c5e39f490a49df14320f4e8c80161cfcce09f1e2cde1edd16a551abb3/frozenlist-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl",
        sha256="494a5952b1c597ba44e0e78113a7266e656b9794eec897b19ead706bd7074383",
    ),
    LockedDistribution(
        name="fsspec",
        version="2025.12.0",
        filename="fsspec-2025.12.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/51/c7/b64cae5dba3a1b138d7123ec36bb5ccd39d39939f18454407e5468f4763f/fsspec-2025.12.0-py3-none-any.whl",
        sha256="8bf1fe301b7d8acfa6e8571e3b1c3d158f909666642431cc78a1b7b4dbc5ec5b",
    ),
    LockedDistribution(
        name="hf-xet",
        version="1.5.1",
        filename="hf_xet-1.5.1-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        url="https://files.pythonhosted.org/packages/de/cc/f99f4bc7295023d7bd9ebbfd51f75cc530ca262c1227666268b8208f4b77/hf_xet-1.5.1-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256="892e3a3a3aecc12aded8b93cf4f9cd059282c7de0732f7d55026f3abdf474350",
    ),
    LockedDistribution(
        name="huggingface-hub",
        version="0.36.2",
        filename="huggingface_hub-0.36.2-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/a8/af/48ac8483240de756d2438c380746e7130d1c6f75802ef22f3c6d49982787/huggingface_hub-0.36.2-py3-none-any.whl",
        sha256="48f0c8eac16145dfce371e9d2d7772854a4f591bcb56c9cf548accf531d54270",
    ),
    LockedDistribution(
        name="idna",
        version="3.11",
        filename="idna-3.11-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/0e/61/66938bbb5fc52dbdf84594873d5b51fb1f7c7794e9c0f5bd885f30bc507b/idna-3.11-py3-none-any.whl",
        sha256="771a87f49d9defaf64091e6e6fe9c18d4833f140bd19464795bc32d966ca37ea",
    ),
    LockedDistribution(
        name="importlib-metadata",
        version="8.7.0",
        filename="importlib_metadata-8.7.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/20/b0/36bd937216ec521246249be3bf9855081de4c5e06a0c9b4219dbeda50373/importlib_metadata-8.7.0-py3-none-any.whl",
        sha256="e5dd1551894c77868a30651cef00984d50e1002d06942a7101d34870c5f02afd",
    ),
    LockedDistribution(
        name="jinja2",
        version="3.1.6",
        filename="jinja2-3.1.6-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl",
        sha256="85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67",
    ),
    LockedDistribution(
        name="kiwisolver",
        version="1.4.9",
        filename="kiwisolver-1.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        url="https://files.pythonhosted.org/packages/70/90/6d240beb0f24b74371762873e9b7f499f1e02166a2d9c5801f4dbf8fa12e/kiwisolver-1.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256="f6008a4919fdbc0b0097089f67a1eb55d950ed7e90ce2cc3e640abadd2757a04",
    ),
    LockedDistribution(
        name="markupsafe",
        version="3.0.3",
        filename="markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        url="https://files.pythonhosted.org/packages/3c/2e/8d0c2ab90a8c1d9a24f0399058ab8519a3279d1bd4289511d74e909f060e/markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        sha256="d6dd0be5b5b189d31db7cda48b91d7e0a9795f31430b7f271219ab30f1d3ac9d",
    ),
    LockedDistribution(
        name="matplotlib",
        version="3.10.8",
        filename="matplotlib-3.10.8-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        url="https://files.pythonhosted.org/packages/3e/f3/c5195b1ae57ef85339fd7285dfb603b22c8b4e79114bae5f4f0fcf688677/matplotlib-3.10.8-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256="3ab4aabc72de4ff77b3ec33a6d78a68227bf1123465887f9905ba79184a1cc04",
    ),
    LockedDistribution(
        name="mpmath",
        version="1.3.0",
        filename="mpmath-1.3.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl",
        sha256="a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c",
    ),
    LockedDistribution(
        name="multidict",
        version="6.7.0",
        filename="multidict-6.7.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        url="https://files.pythonhosted.org/packages/0d/e2/9baffdae21a76f77ef8447f1a05a96ec4bc0a24dae08767abc0a2fe680b8/multidict-6.7.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        sha256="123e2a72e20537add2f33a79e605f6191fba2afda4cbb876e35c1a7074298a7d",
    ),
    LockedDistribution(
        name="networkx",
        version="3.4.2",
        filename="networkx-3.4.2-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/b9/54/dd730b32ea14ea797530a4479b2ed46a6fb250f682a9cfb997e968bf0261/networkx-3.4.2-py3-none-any.whl",
        sha256="df5d4365b724cf81b8c6a7312509d0c22386097011ad1abe274afd5e9d3bbc5f",
    ),
    LockedDistribution(
        name="numpy",
        version="1.26.4",
        filename="numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        url="https://files.pythonhosted.org/packages/0f/50/de23fde84e45f5c4fda2488c759b69990fd4512387a8632860f3ac9cd225/numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        sha256="675d61ffbfa78604709862923189bad94014bef562cc35cf61d3a07bba02a7ed",
    ),
    LockedDistribution(
        name="nvidia-cublas-cu12",
        version="12.8.4.1",
        filename="nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl",
        url="https://files.pythonhosted.org/packages/dc/61/e24b560ab2e2eaeb3c839129175fb330dfcfc29e5203196e5541a4c44682/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl",
        sha256="8ac4e771d5a348c551b2a426eda6193c19aa630236b418086020df5ba9667142",
    ),
    LockedDistribution(
        name="nvidia-cuda-cupti-cu12",
        version="12.8.90",
        filename="nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        url="https://files.pythonhosted.org/packages/f8/02/2adcaa145158bf1a8295d83591d22e4103dbfd821bcaf6f3f53151ca4ffa/nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256="ea0cb07ebda26bb9b29ba82cda34849e73c166c18162d3913575b0c9db9a6182",
    ),
    LockedDistribution(
        name="nvidia-cuda-nvrtc-cu12",
        version="12.8.93",
        filename="nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        url="https://files.pythonhosted.org/packages/05/6b/32f747947df2da6994e999492ab306a903659555dddc0fbdeb9d71f75e52/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        sha256="a7756528852ef889772a84c6cd89d41dfa74667e24cca16bb31f8f061e3e9994",
    ),
    LockedDistribution(
        name="nvidia-cuda-runtime-cu12",
        version="12.8.90",
        filename="nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        url="https://files.pythonhosted.org/packages/0d/9b/a997b638fcd068ad6e4d53b8551a7d30fe8b404d6f1804abf1df69838932/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256="adade8dcbd0edf427b7204d480d6066d33902cab2a4707dcfc48a2d0fd44ab90",
    ),
    LockedDistribution(
        name="nvidia-cudnn-cu12",
        version="9.19.0.56",
        filename="nvidia_cudnn_cu12-9.19.0.56-py3-none-manylinux_2_27_x86_64.whl",
        url="https://files.pythonhosted.org/packages/c5/41/65225d42fba06fb3dd3972485ea258e7dd07a40d6e01c95da6766ad87354/nvidia_cudnn_cu12-9.19.0.56-py3-none-manylinux_2_27_x86_64.whl",
        sha256="ac6ad90a075bb33a94f2b4cf4622eac13dd4dc65cf6dd9c7572a318516a36625",
    ),
    LockedDistribution(
        name="nvidia-cufft-cu12",
        version="11.3.3.83",
        filename="nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        url="https://files.pythonhosted.org/packages/1f/13/ee4e00f30e676b66ae65b4f08cb5bcbb8392c03f54f2d5413ea99a5d1c80/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256="4d2dd21ec0b88cf61b62e6b43564355e5222e4a3fb394cac0db101f2dd0d4f74",
    ),
    LockedDistribution(
        name="nvidia-cufile-cu12",
        version="1.13.1.3",
        filename="nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        url="https://files.pythonhosted.org/packages/bb/fe/1bcba1dfbfb8d01be8d93f07bfc502c93fa23afa6fd5ab3fc7c1df71038a/nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256="1d069003be650e131b21c932ec3d8969c1715379251f8d23a1860554b1cb24fc",
    ),
    LockedDistribution(
        name="nvidia-curand-cu12",
        version="10.3.9.90",
        filename="nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl",
        url="https://files.pythonhosted.org/packages/fb/aa/6584b56dc84ebe9cf93226a5cde4d99080c8e90ab40f0c27bda7a0f29aa1/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl",
        sha256="b32331d4f4df5d6eefa0554c565b626c7216f87a06a4f56fab27c3b68a830ec9",
    ),
    LockedDistribution(
        name="nvidia-cusolver-cu12",
        version="11.7.3.90",
        filename="nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl",
        url="https://files.pythonhosted.org/packages/85/48/9a13d2975803e8cf2777d5ed57b87a0b6ca2cc795f9a4f59796a910bfb80/nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl",
        sha256="4376c11ad263152bd50ea295c05370360776f8c3427b30991df774f9fb26c450",
    ),
    LockedDistribution(
        name="nvidia-cusparse-cu12",
        version="12.5.8.93",
        filename="nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        url="https://files.pythonhosted.org/packages/c2/f5/e1854cb2f2bcd4280c44736c93550cc300ff4b8c95ebe370d0aa7d2b473d/nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256="1ec05d76bbbd8b61b06a80e1eaf8cf4959c3d4ce8e711b65ebd0443bb0ebb13b",
    ),
    LockedDistribution(
        name="nvidia-cusparselt-cu12",
        version="0.7.1",
        filename="nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl",
        url="https://files.pythonhosted.org/packages/56/79/12978b96bd44274fe38b5dde5cfb660b1d114f70a65ef962bcbbed99b549/nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl",
        sha256="f1bb701d6b930d5a7cea44c19ceb973311500847f81b634d802b7b539dc55623",
    ),
    LockedDistribution(
        name="nvidia-nccl-cu12",
        version="2.28.9",
        filename="nvidia_nccl_cu12-2.28.9-py3-none-manylinux_2_18_x86_64.whl",
        url="https://files.pythonhosted.org/packages/4a/4e/44dbb46b3d1b0ec61afda8e84837870f2f9ace33c564317d59b70bc19d3e/nvidia_nccl_cu12-2.28.9-py3-none-manylinux_2_18_x86_64.whl",
        sha256="485776daa8447da5da39681af455aa3b2c2586ddcf4af8772495e7c532c7e5ab",
    ),
    LockedDistribution(
        name="nvidia-nvjitlink-cu12",
        version="12.8.93",
        filename="nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        url="https://files.pythonhosted.org/packages/f6/74/86a07f1d0f42998ca31312f998bd3b9a7eff7f52378f4f270c8679c77fb9/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        sha256="81ff63371a7ebd6e6451970684f916be2eab07321b73c9d244dc2b4da7f73b88",
    ),
    LockedDistribution(
        name="nvidia-nvshmem-cu12",
        version="3.4.5",
        filename="nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        url="https://files.pythonhosted.org/packages/b5/09/6ea3ea725f82e1e76684f0708bbedd871fc96da89945adeba65c3835a64c/nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256="042f2500f24c021db8a06c5eec2539027d57460e1c1a762055a6554f72c369bd",
    ),
    LockedDistribution(
        name="nvidia-nvtx-cu12",
        version="12.8.90",
        filename="nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        url="https://files.pythonhosted.org/packages/a2/eb/86626c1bbc2edb86323022371c39aa48df6fd8b0a1647bc274577f72e90b/nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256="5b17e2001cc0d751a5bc2c6ec6d26ad95913324a4adb86788c944f8ce9ba441f",
    ),
    LockedDistribution(
        name="omegaconf",
        version="2.3.0",
        filename="omegaconf-2.3.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/e3/94/1843518e420fa3ed6919835845df698c7e27e183cb997394e4a670973a65/omegaconf-2.3.0-py3-none-any.whl",
        sha256="7b4df175cdb08ba400f45cae3bdcae7ba8365db4d165fc65fd04b050ab63b46b",
    ),
    LockedDistribution(
        name="opencv-python",
        version="4.9.0.80",
        filename="opencv_python-4.9.0.80-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        url="https://files.pythonhosted.org/packages/d9/64/7fdfb9386511cd6805451e012c537073a79a958a58795c4e602e538c388c/opencv_python-4.9.0.80-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        sha256="e4088cab82b66a3b37ffc452976b14a3c599269c247895ae9ceb4066d8188a57",
    ),
    LockedDistribution(
        name="packaging",
        version="24.2",
        filename="packaging-24.2-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/88/ef/eb23f262cca3c0c4eb7ab1933c3b1f03d021f2c48f54763065b6f0e321be/packaging-24.2-py3-none-any.whl",
        sha256="09abb1bccd265c01f4a3aa3f7a7db064b36514d2cba19a2f694fe6150451a759",
    ),
    LockedDistribution(
        name="pillow",
        version="11.1.0",
        filename="pillow-11.1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        url="https://files.pythonhosted.org/packages/38/0d/84200ed6a871ce386ddc82904bfadc0c6b28b0c0ec78176871a4679e40b3/pillow-11.1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        sha256="9aa9aeddeed452b2f616ff5507459e7bab436916ccb10961c4a382cd3e03f47f",
    ),
    LockedDistribution(
        name="platformdirs",
        version="4.3.7",
        filename="platformdirs-4.3.7-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/6d/45/59578566b3275b8fd9157885918fcd0c4d74162928a5310926887b856a51/platformdirs-4.3.7-py3-none-any.whl",
        sha256="a03875334331946f13c549dbd8f4bac7a13a50a895a0eb1e8c6a8ace80d40a94",
    ),
    LockedDistribution(
        name="propcache",
        version="0.4.1",
        filename="propcache-0.4.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        url="https://files.pythonhosted.org/packages/46/4b/3aae6835b8e5f44ea6a68348ad90f78134047b503765087be2f9912140ea/propcache-0.4.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        sha256="15932ab57837c3368b024473a525e25d316d8353016e7cc0e5ba9eb343fbb1cf",
    ),
    LockedDistribution(
        name="pycocotools",
        version="2.0.11",
        filename="pycocotools-2.0.11-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        url="https://files.pythonhosted.org/packages/23/59/dc81895beff4e1207a829d40d442ea87cefaac9f6499151965f05c479619/pycocotools-2.0.11-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        sha256="a82d1c9ed83f75da0b3f244f2a3cf559351a283307bd9b79a4ee2b93ab3231dd",
    ),
    LockedDistribution(
        name="pyparsing",
        version="3.2.5",
        filename="pyparsing-3.2.5-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/10/5e/1aa9a93198c6b64513c9d7752de7422c06402de6600a8767da1524f9570b/pyparsing-3.2.5-py3-none-any.whl",
        sha256="e38a4f02064cf41fe6593d328d0512495ad1f3d8a91c4f73fc401b3079a59a5e",
    ),
    LockedDistribution(
        name="python-dateutil",
        version="2.9.0.post0",
        filename="python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/ec/57/56b9bcc3c9c6a792fcbaf139543cee77261f3651ca9da0c93f5c1221264b/python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
        sha256="a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427",
    ),
    LockedDistribution(
        name="pyyaml",
        version="6.0.2",
        filename="PyYAML-6.0.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        url="https://files.pythonhosted.org/packages/b9/2b/614b4752f2e127db5cc206abc23a8c19678e92b23c3db30fc86ab731d3bd/PyYAML-6.0.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        sha256="80bab7bfc629882493af4aa31a4cfa43a4c57c83813253626916b8c7ada83476",
    ),
    LockedDistribution(
        name="regex",
        version="2025.11.3",
        filename="regex-2025.11.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        url="https://files.pythonhosted.org/packages/84/bd/9ce9f629fcb714ffc2c3faf62b6766ecb7a585e1e885eb699bcf130a5209/regex-2025.11.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        sha256="a12ab1f5c29b4e93db518f5e3872116b7e9b1646c9f9f426f777b50d44a09e8c",
    ),
    LockedDistribution(
        name="requests",
        version="2.32.4",
        filename="requests-2.32.4-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/7c/e4/56027c4a6b4ae70ca9de302488c5ca95ad4a39e190093d6c1a8ace08341b/requests-2.32.4-py3-none-any.whl",
        sha256="27babd3cda2a6d50b30443204ee89830707d396671944c998b5975b031ac2b2c",
    ),
    LockedDistribution(
        name="safetensors",
        version="0.8.0",
        filename="safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        url="https://files.pythonhosted.org/packages/28/50/f203ff3a3ddfe19308efc83c5a3a29ed02bf786732ec35e68bf9162f3365/safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        sha256="fd6f3f93c9a0a7cc2788ee63fb763353d4bd2e89b0751bc78fcf7dda00bea774",
    ),
    LockedDistribution(
        name="scipy",
        version="1.17.1",
        filename="scipy-1.17.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        url="https://files.pythonhosted.org/packages/01/8e/1e35281b8ab6d5d72ebe9911edcdffa3f36b04ed9d51dec6dd140396e220/scipy-1.17.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        sha256="02ae3b274fde71c5e92ac4d54bc06c42d80e399fec704383dcd99b301df37458",
    ),
    LockedDistribution(
        name="setuptools",
        version="78.1.1",
        filename="setuptools-78.1.1-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/90/99/158ad0609729111163fc1f674a5a42f2605371a4cf036d0441070e2f7455/setuptools-78.1.1-py3-none-any.whl",
        sha256="c3a9c4211ff4c309edb8b8c4f1cbfa7ae324c4ba9f91ff254e3d305b9fd54561",
    ),
    LockedDistribution(
        name="six",
        version="1.17.0",
        filename="six-1.17.0-py2.py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl",
        sha256="4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274",
    ),
    LockedDistribution(
        name="sympy",
        version="1.14.0",
        filename="sympy-1.14.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl",
        sha256="e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5",
    ),
    LockedDistribution(
        name="termcolor",
        version="3.3.0",
        filename="termcolor-3.3.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/33/d1/8bb87d21e9aeb323cc03034f5eaf2c8f69841e40e4853c2627edf8111ed3/termcolor-3.3.0-py3-none-any.whl",
        sha256="cf642efadaf0a8ebbbf4bc7a31cec2f9b5f21a9f726f4ccbb08192c9c26f43a5",
    ),
    LockedDistribution(
        name="timm",
        version="0.9.16",
        filename="timm-0.9.16-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/68/99/2018622d268f6017ddfa5ee71f070bad5d07590374793166baa102849d17/timm-0.9.16-py3-none-any.whl",
        sha256="bf5704014476ab011589d3c14172ee4c901fd18f9110a928019cac5be2945914",
    ),
    LockedDistribution(
        name="tokenizers",
        version="0.21.4",
        filename="tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        url="https://files.pythonhosted.org/packages/f2/90/273b6c7ec78af547694eddeea9e05de771278bd20476525ab930cecaf7d8/tokenizers-0.21.4-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        sha256="51b7eabb104f46c1c50b486520555715457ae833d5aee9ff6ae853d1130506ff",
    ),
    LockedDistribution(
        name="tomli",
        version="2.4.1",
        filename="tomli-2.4.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        url="https://files.pythonhosted.org/packages/10/90/d62ce007a1c80d0b2c93e02cab211224756240884751b94ca72df8a875ca/tomli-2.4.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        sha256="136443dbd7e1dee43c68ac2694fde36b2849865fa258d39bf822c10e8068eac5",
    ),
    LockedDistribution(
        name="torch",
        version="2.11.0+cu128",
        filename="torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl",
        url="https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl",
        sha256="d252cf975fb18c94a85336323ad425f473df56dab35a44b00399bd70c7a3b997",
        source=ArtifactSource.PYTORCH,
    ),
    LockedDistribution(
        name="torchvision",
        version="0.26.0+cu128",
        filename="torchvision-0.26.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl",
        url="https://download-r2.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl",
        sha256="ccf26b4b659cfce6f2208cb8326071d51c70219a34856dfdf468d1e19af52c0d",
        source=ArtifactSource.PYTORCH,
    ),
    LockedDistribution(
        name="tqdm",
        version="4.66.5",
        filename="tqdm-4.66.5-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/48/5d/acf5905c36149bbaec41ccf7f2b68814647347b72075ac0b1fe3022fdc73/tqdm-4.66.5-py3-none-any.whl",
        sha256="90279a3770753eafc9194a0364852159802111925aa30eb3f9d85b0e805ac7cd",
    ),
    LockedDistribution(
        name="transformers",
        version="4.48.3",
        filename="transformers-4.48.3-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/b6/1a/efeecb8d83705f2f4beac98d46f2148c95ecd7babfb31b5c0f1e7017e83d/transformers-4.48.3-py3-none-any.whl",
        sha256="78697f990f5ef350c23b46bf86d5081ce96b49479ab180b2de7687267de8fd36",
    ),
    LockedDistribution(
        name="triton",
        version="3.6.0",
        filename="triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        url="https://download-r2.pytorch.org/whl/triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        sha256="6f5928e6d44c34a97bbe164cceddc0ef2007121c89ebcfba5415cf452de7ee9f",
        source=ArtifactSource.PYTORCH,
    ),
    LockedDistribution(
        name="typing-extensions",
        version="4.15.0",
        filename="typing_extensions-4.15.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/18/67/36e9267722cc04a6b9f15c7f3441c2363321a3ea07da7ae0c0707beb2a9c/typing_extensions-4.15.0-py3-none-any.whl",
        sha256="f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548",
    ),
    LockedDistribution(
        name="urllib3",
        version="2.6.3",
        filename="urllib3-2.6.3-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/39/08/aaaad47bc4e9dc8c725e68f9d04865dbcb2052843ff09c97b08904852d84/urllib3-2.6.3-py3-none-any.whl",
        sha256="bf272323e553dfb2e87d9bfd225ca7b0f467b919d7bbd355436d3fd37cb0acd4",
    ),
    LockedDistribution(
        name="yapf",
        version="0.40.1",
        filename="yapf-0.40.1-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/23/75/c374517c09e31bf22d3b3f156d73e0f38d08e29b2afdd607cef5f1e10aa9/yapf-0.40.1-py3-none-any.whl",
        sha256="b8bfc1f280949153e795181768ca14ef43d7312629a06c43e7abd279323af313",
    ),
    LockedDistribution(
        name="yarl",
        version="1.22.0",
        filename="yarl-1.22.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        url="https://files.pythonhosted.org/packages/db/0f/0d52c98b8a885aeda831224b78f3be7ec2e1aa4a62091f9f9188c3c65b56/yarl-1.22.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
        sha256="50678a3b71c751d58d7908edc96d332af328839eea883bb554a43f539101277a",
    ),
    LockedDistribution(
        name="zipp",
        version="3.23.0",
        filename="zipp-3.23.0-py3-none-any.whl",
        url="https://files.pythonhosted.org/packages/2e/54/647ade08bf0db230bfea292f893923872fd20be6ac6f53b2b936ba839d75/zipp-3.23.0-py3-none-any.whl",
        sha256="071652d6115ed432f5ce1d34c336c0adfd6a884660d1e9712a256d3d3bd4b14e",
    ),
)


def locked_package_versions() -> dict[str, str]:
    """Return runtime distributions without the separate pip bootstrap."""

    versions: dict[str, str] = {}
    for distribution in ENVIRONMENT_LOCK:
        if distribution.name in versions:
            raise ValueError(
                f"duplicate CountGD lock package: {distribution.name}",
            )
        versions[distribution.name] = distribution.version
    return versions


def expected_environment_package_versions() -> dict[str, str]:
    """Return the exact distribution set required in the isolated venv."""

    return {"pip": QUALIFIED_PIP_VERSION, **locked_package_versions()}


def environment_lock_payload() -> dict[str, Any]:
    """Return the manifest/report representation used to derive lock identity."""

    return {
        "schema": ENVIRONMENT_LOCK_SCHEMA,
        "target": {
            "python": QUALIFIED_PYTHON_VERSION,
            "platform": QUALIFIED_PLATFORM,
            "machine": QUALIFIED_MACHINE,
        },
        "bootstrap": {"pip": QUALIFIED_PIP_VERSION},
        "package_count": len(ENVIRONMENT_LOCK),
        "artifacts": [
            {
                "name": distribution.name,
                "version": distribution.version,
                "filename": distribution.filename,
                "url": distribution.url,
                "sha256": distribution.sha256,
                "source": distribution.source.value,
                "artifact_kind": distribution.artifact_kind.value,
            }
            for distribution in ENVIRONMENT_LOCK
        ],
    }


def environment_lock_digest() -> str:
    """Return a stable digest over the target, bootstrap pin, and artifacts."""

    encoded = json.dumps(
        environment_lock_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
