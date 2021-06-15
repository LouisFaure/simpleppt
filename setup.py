from setuptools import setup, find_packages

setup(
    name="SimplePPT",
    version_format="{tag}",
    setup_requires=["setuptools-git-version"],
    description="Python implementation of SimplePPT algorithm, with GPU acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LouisFaure/SimplePPT",
    author="Louis Faure",
    author_email="",
    packages=find_packages(),
    package_dir={"SimplePPT": "SimplePPT"},
    install_requires=requirements,
    zip_safe=False,
)