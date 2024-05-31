"""Sets up the current directory as a python package for easier imports

This is used in conjunction with "pip install -e ."
"""

from setuptools import setup, find_packages

setup(
    name="reachbot_manipulation",
    version="0.0.1",
    install_requires=[
        "numpy",
        "matplotlib",
        "tqdm",
        "scipy",
        "cvxpy",
        "pytransform3d",
        "clarabel",
        "mosek",
        "pybullet",
        "ipython",
        "wheel",
        "pyyaml",
    ],
    extras_require={"dev": ["pylint", "black"]},
    description="Manipulation with Reachbot",
    author="Daniel Morton",
    author_email="danielpmorton@gmail.com",
    url="https://github.com/StanfordASL/reachbot_manipulation",
    packages=find_packages(exclude=["data", "artifacts", "images"]),
)


# Tested to work in the following environment, with python 3.10.8
# Package                  Version
# ------------------------ -----------
# absl-py                  2.0.0
# astroid                  2.15.7
# async-timeout            4.0.3
# black                    23.9.1
# clarabel                 0.6.0
# click                    8.1.7
# cloudpickle              3.0.0
# cmake                    3.27.6
# contourpy                1.1.1
# cvxpy                    1.3.2
# cycler                   0.11.0
# dill                     0.3.7
# docstring-parser         0.15
# ecos                     2.0.12
# etils                    1.5.1
# filelock                 3.12.4
# fonttools                4.42.1
# fsspec                   2023.9.2
# glfw                     2.6.2
# importlib-resources      6.1.0
# ipyopt                   0.12.7
# isort                    5.12.0
# jax                      0.4.8
# jax-dataclasses          1.5.1
# jaxlib                   0.4.7
# jaxlie                   1.3.3
# Jinja2                   3.1.2
# julia                    0.6.1
# kiwisolver               1.4.5
# lazy-object-proxy        1.9.0
# lit                      17.0.1
# lxml                     4.9.3
# markdown-it-py           3.0.0
# MarkupSafe               2.1.3
# matplotlib               3.8.0
# mccabe                   0.7.0
# mdurl                    0.1.2
# ml-dtypes                0.3.1
# Mosek                    10.1.17
# mpmath                   1.3.0
# mujoco                   3.0.0
# mypy                     1.5.1
# mypy-extensions          1.0.0
# networkx                 3.1
# ninja                    1.11.1
# numpy                    1.26.0
# nvidia-cublas-cu11       11.10.3.66
# nvidia-cuda-cupti-cu11   11.7.101
# nvidia-cuda-nvrtc-cu11   11.7.99
# nvidia-cuda-runtime-cu11 11.7.99
# nvidia-cudnn-cu11        8.5.0.96
# nvidia-cufft-cu11        10.9.0.58
# nvidia-curand-cu11       10.2.10.91
# nvidia-cusolver-cu11     11.4.0.1
# nvidia-cusparse-cu11     11.7.4.91
# nvidia-nccl-cu11         2.14.3
# nvidia-nvtx-cu11         11.7.91
# opt-einsum               3.3.0
# osqp                     0.6.3
# packaging                23.1
# pathspec                 0.11.2
# Pillow                   10.0.1
# pip                      22.2.2
# platformdirs             3.10.0
# pmpc                     0.7.1
# psutil                   5.9.6
# pybullet                 3.2.5
# Pygments                 2.16.1
# pylint                   2.17.6
# PyOpenGL                 3.1.7
# pyparsing                3.1.1
# python-dateutil          2.8.2
# pytransform3d            3.4.0
# PyYAML                   6.0.1
# pyzmq                    25.1.1
# qdldl                    0.1.7.post0
# redis                    5.0.1
# rich                     13.6.0
# scipy                    1.11.2
# scs                      3.2.3
# setuptools               68.2.2
# shtab                    1.6.4
# six                      1.16.0
# sympy                    1.12
# tomli                    2.0.1
# tomlkit                  0.12.1
# torch                    2.0.1
# torch2jax                0.4.6
# tqdm                     4.66.1
# triton                   2.0.0
# typing_extensions        4.8.0
# tyro                     0.5.9
# wheel                    0.41.2
# wrapt                    1.15.0
# zipp                     3.17.0
# zstandard                0.21.0
