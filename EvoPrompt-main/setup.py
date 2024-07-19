#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="inclearn",
    version="0.1",
    description="An Incremental Learning library.",
    author=["Arthur Douillard", "Muhammad Rifki Kurniawan"],
    url="https://github.com/arthurdouillard/incremental_learning.pytorch",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={"console_scripts": ["inclearn = inclearn.__main__:main"]},
)
