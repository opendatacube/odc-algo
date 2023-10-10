from setuptools import setup, find_packages
from setuptools_rust import RustExtension, Strip


# Never build rust debug version, it's too slow to do anything
setup(
    packages=find_packages(),
    rust_extensions=[RustExtension("odc.algo.backend", 
                                   strip=Strip.No,
                                   debug=False)]
)
