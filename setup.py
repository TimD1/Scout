import os
import re
from setuptools import setup, find_packages

# set package name
__pkg_name__ = 'scout'

# parse project version from package init file
verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))

# parse project dependencies
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# parse project description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# set up package
setup(
    name=__pkg_name__,
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Tim Dunn',
    author_email='me.timd1@gmail.com',
    url='https://github.com/TimD1/Scout',
    entry_points = {
        'console_scripts': [
            '{0} = {0}:main'.format(__pkg_name__)
        ]
    },
)
