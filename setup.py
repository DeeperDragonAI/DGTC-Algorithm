from setuptools import setup, find_packages

setup(
    name="dgtc-framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'pywavelets>=1.4.0',
    ],
    python_requires='>=3.8',
)