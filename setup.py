from setuptools import setup, find_packages
from setuptools_rust import RustExtension, Strip


setup(
    packages=find_packages(),
    rust_extensions=[RustExtension("odc.algo.backend", strip=Strip.No)]
)
