import os
import re

import setuptools

directory = os.path.dirname(os.path.abspath(__file__))

# Extract version information
path = os.path.join(directory, 'goggles', '__init__.py')
with open(path) as read_file:
    text = read_file.read()
pattern = re.compile(r"^__version__ = ['\"]([^'\"]*)['\"]", re.MULTILINE)
version = pattern.search(text).group(1)

# Extract long_description
path = os.path.join(directory, 'README.md')
with open(path) as read_file:
    long_description = read_file.read()

setuptools.setup(
    name='goggles',
    version=version,
    url='https://github.com/chu-data-lab/GOGGLES',
    description='A system for automatically generating probabilistic labels for image datasets based on the affinity coding paradigm',
    long_description_content_type='text/markdown',
    long_description=long_description,
    license='MIT License',
    packages=setuptools.find_packages(),
    include_package_data=True,

    keywords='traning-data-generation machine-learning ai information-extraction affinity-coding',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],

    project_urls={  # Optional
        'Homepage': 'https://github.com/chu-data-lab/GOGGLES',
        'Source': 'https://github.com/chu-data-lab/GOGGLES',
        'Bug Reports': 'https://github.com/chu-data-lab/GOGGLES/issues',
        'Citation': 'https://arxiv.org/abs/1903.04552',
    },
)