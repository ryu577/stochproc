from setuptools import setup, find_packages, Extension

setup(name='stochproc',
      version='0.0.7',
      url='https://github.com/ryu577/stochproc',
      license='MIT',
      author='Rohit Pandey',
      author_email='rohitpandey576@gmail.com',
      description='Methods to model all kinds of stochastic processes.',
      packages=find_packages(exclude=['tests','plots']),
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      #long_description='Modeling all kinds of stochstic processes. Coin toss sequences, failure rates and most recently, hypothesis testing.',
      zip_safe=False)

