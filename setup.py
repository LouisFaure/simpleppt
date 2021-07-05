from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="simpleppt",
    version_config={
        "template": "{tag}",
        "dev_template": "{tag}",
        "dirty_template": "{tag}",
    },
    setup_requires=["setuptools-git-versioning"],
    description="Python implementation of SimplePPT algorithm, with GPU acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LouisFaure/SimplePPT",
    author="Louis Faure",
    author_email="",
    include_package_data=True,
    package_data={"": ["tests/*.csv"]},
    packages=find_packages(),
    package_dir={"simpleppt": "simpleppt"},
    install_requires=requirements,
    zip_safe=False,
)
