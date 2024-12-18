from setuptools import setup, find_packages

setup(
    name='fsinc',
    version='0.2.0',
    url='https://github.com/gauteh/fsinc.git',
    author='Gaute Hope',
    author_email='gauteh@met.no',
    description='Implementation of fast sinc-interpolation and sinc-transform based on FINUFFT and https://github.com/hannahlawrence/sinctransform',
    packages=find_packages(),
    install_requires=['numpy', 'numba', 'scipy', 'matplotlib', 'finufft'],
)

