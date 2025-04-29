from setuptools import setup, find_packages

setup(
    name="crypto-aml-risk-detection",
    version="0.1",
    packages=find_packages(),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'networkx',
        'matplotlib',
        'seaborn',
        'joblib'
    ],
) 