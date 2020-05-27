from setuptools import setup, find_packages

setup(
    name='fsinc',
    version='0.1.0',
    url='https://github.com/gauteh/fsinc.git',
    author='Gaute Hope',
    author_email='gauteh@met.no',
    description='Implementation of fast sinc-transform based on https://github.com/hannahlawrence/sinctransform',
    packages=find_packages(),
    install_requires=['numpy', 'numba', 'scipy', 'matplotlib', 'finufftpy'],
)

