from setuptools import setup, find_packages

setup(name='stochproc',
      version='0.0.0',
      url='https://github.com/ryu577/stochproc',
      license='MIT',
      author='Rohit Pandey',
      author_email='rohitpandey576@gmail.com',
      description='Methods to model all kinds of stochastic processes.',
      packages=find_packages(exclude=['tests']),
      long_description=open('README.md').read(),
      zip_safe=False)

