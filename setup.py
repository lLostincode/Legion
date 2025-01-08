from setuptools import setup, find_packages

setup(
    name="legion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "openai>=1.0.0"
    ]
)
