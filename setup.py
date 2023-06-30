from setuptools import setup, find_packages

setup(
    name='deep-autoencoder-for-shape-optimization',
    version='1.0',
    url='https://github.com/lekos21/dae',
    packages=find_packages(),
    author='lekos',
    description='Deep autoencoder package',
    long_description='A deep autoencoder package for shape optimization',
    download_url='https://github.com/lekos21/dae',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.18.0",
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ]
)
