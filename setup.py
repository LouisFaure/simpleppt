from setuptools import setup, find_packages
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="simpleppt",
    version_format="{tag}",
    setup_requires=["setuptools-git-version"],
    description="Python implementation of SimplePPT algorithm, with GPU acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LouisFaure/SimplePPT",
    author="Louis Faure",
    author_email="",
    packages=find_packages(),
    package_dir={"simpleppt": "simpleppt"},
    install_requires=requirements,
    zip_safe=False,
)