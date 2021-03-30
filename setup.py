import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

# Dependencies (format is 'PYPI_PACKAGE_NAME[>=]=VERSION_NUMBER')
BASE_DEPENDENCIES = [
    'boto3>=1.17',
    'cachier>=1.5.0',
    'click>=7.1.1',
    'click-log>=0.3.2',
    'keras>=2.4.3',
    'matplotlib>=3.3.3',
    'nocasedict>=1.0.2',
    'numpy~=1.20.1',
    'pandas>=1.1.4',
    'python-dotenv>=0.14.0',
    'python-slugify>=4.0.0',
    'scikit-learn>=0.24',
    'scipy>=1.5.4',
    #'tensorflow>=2.4.1',
    'wf-honeycomb-io>=1.3.0',
    'wf-geom-render>=0.3.0'
]
# TEST_DEPENDENCIES = [
# ]
#
DEVELOPMENT_DEPENDENCIES = [
    'autopep8>=1.5.2',
    'pytest>=6.2.2'
]

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(BASEDIR))

setup(
    name='wf-process-cuwb-data',
    packages=find_packages(),
    version=VERSION,
    include_package_data=True,
    description='Tools for reading, processing, and writing CUWB data',
    long_description=open('README.md').read(),
    url='https://github.com/WildflowerSchools/wf-process-cuwb-data',
    author='Theodore Quinn',
    author_email='ted.quinn@wildflowerschools.org',
    install_requires=BASE_DEPENDENCIES,
    # tests_require=TEST_DEPENDENCIES,
    extras_require={
        'development': DEVELOPMENT_DEPENDENCIES
    },
    entry_points={
        "console_scripts": [
             "process_cuwb_data = process_cuwb_data.cli:cli"
        ]
    },
    # keywords=['KEYWORD'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
