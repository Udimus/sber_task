import os
from setuptools import setup, find_packages

setup_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(setup_dir, "requirements.txt"), "r") as req_file:
    requirements = [line.strip() for line in req_file if line.strip()]

setup(
    name="module",
    version="0.1",
    install_requires=requirements,
    packages=find_packages(include=["module"]),
)
