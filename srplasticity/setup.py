from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='srplasticity',
      version='0.1',
      description='Functions for linear-nonlinear synaptic convolution model',
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
