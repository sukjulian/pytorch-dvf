from setuptools import find_packages, setup

setup(
    name="torch_dvf",
    version="0.3.0",
    author="Julian Suk",
    author_email="j.m.suk@utwente.nl",
    license="MIT",
    packages=find_packages(),
    install_requires=["torch", "torch_geometric", "torch_scatter", "torch_cluster"],
)
