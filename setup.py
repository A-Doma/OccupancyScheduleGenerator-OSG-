from setuptools import setup, find_packages

setup(
    name='OSG',
    version='0.1',
    packages=find_packages(),
    description='Occupancy Schedule Generator from DYD data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Aya Doma - IBCL',
    author_email='aya.doma@mail.concordia.com',
    url='https://github.com/AyaDoma/OccupancyScheduleGenerator-OSG-',
    install_requires=[
        # List your package dependencies here
        'Pandas',
        'ipywidgets',
        'numpy'
    ],
)
