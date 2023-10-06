from setuptools import find_packages, setup

setup(
    name="ndeep",
    version="1.0.0",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    description="Research Library for the nDeep project",
    author="Ngoc-Dung Nguyen",
    author_email="nguyenn@moadata.ai.kr",
    license="MIT",
    packages=find_packages(),
)
