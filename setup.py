"""
Jsrplasticity package setup file 
Copyright (C) 2021 Julian Rossbroich, Daniel Trotter, John Beninger, Richard Naud

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='srplasticity',
      version='0.0.1',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Programming Language :: Python :: 3.8',
          ],
      url='https://github.com/nauralcodinglab/srplasticity.git',
      author='Julian Rossbroich, Daniel Trotter, John Beninger, Richard Naud',
      author_email='jbeni014@uottawa.ca',
      license='GPLv3',
      packages=['srplasticity'],
      install_requires=[
          'numpy',
          'scipy',
          ],
      zip_safe=False)
