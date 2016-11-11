from setuptools import setup

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
with open(path.join(here,'h5batchreader','__init__.py'), encoding='utf-8') as f:
    version=f.read().split('__version__=')[1].split()[0].strip()

setup(
    name='h5-batch-reader',
    version=version,
    description='package to prepare batches from h5 input for a machine learning framework.',
    long_description=long_description,
    url='https://github.com/slaclab/h5-batch-reader',
    author='David Schneider',
    author_email='davidsch@slac.stanford.edu',
    license='Stanford',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Machine Learning Developers',
        'Topic :: Software Development :: Machine Learning',
        'License :: Stanford',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='machine learning tensorflow theano hdf5 h5py',
    packages=['h5batchreader'],
    install_requires=['numpy', 'h5py'],
    test_suite='nose.collector',
    tests_require=['nose'],
)
