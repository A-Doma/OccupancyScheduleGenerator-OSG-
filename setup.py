from setuptools import setup, find_packages

setup(
    name='main',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas>=3.0.0',  # Specify the minimum version of pandas required
        'ipywidgets>=7.0.0',  # Specify the minimum version of ipywidgets required
    ],
    author= 'IBCL'
)
