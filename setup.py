from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='maco',
    version='0.0.1',
    author='Jannis Becktepe',
    description='Multi-attribute and Order Indexes for Data Discovery',
    url='https://github.com/LUH-DBS/datalake_indexes',
    keywords='data discovery, data lake',
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'tqdm',
        'sklearn',
        'simhash'
        # TODO fill
    ],
    packages=find_packages(
        exclude=('datasts', 'demonstration', 'fig', 'temp_data'),
        include='maco'
    )
)
