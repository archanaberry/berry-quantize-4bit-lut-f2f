from setuptools import setup, find_packages

setup(
    name="berrybits",
    version="0.2.0",
    description="BerryBits native quantization library and Python API",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
)
