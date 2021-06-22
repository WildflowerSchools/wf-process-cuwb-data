import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

# Dependencies (format is 'PYPI_PACKAGE_NAME[>=]=VERSION_NUMBER')
BASE_DEPENDENCIES = [
    'boto3>=1.17',
    'click>=8.0.0',
    'click-log>=0.3.2',
    'keras>=2.4.3',
    'matplotlib>=3.4.1',
    'nocasedict>=1.0.2',
    'numpy~=1.20.2',
    'pandas>=1.2.4',
    'python-dotenv>=0.17.0',
    'python-slugify>=4.0.0',
    'scikit-learn>=0.24',
    'scipy>=1.6.3',
    #'tensorflow>=2.4.1',
    'wf-honeycomb-io>=1.6.1',
    'wf-geom-render>=0.3.0',
    'wf-process-pose-data>=3.2.1'
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
    long_description_content_type='text/markdown',
    url='https://github.com/WildflowerSchools/wf-process-cuwb-data',
    author='Theodore Quinn',
    author_email='ted.quinn@wildflowerschools.org',
    setup_requires=['numpy'],
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
