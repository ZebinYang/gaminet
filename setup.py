from setuptools import setup

setup(name='gamixnn',
      version='0.1',
      description='Explainable Neural Networks based on Generalized Additive Models with Structured Interactions',
      url='https://github.com/ZebinYang/GAMIxNN',
      author='Zebin Yang',
      author_email='yangzb2010@hku.hk',
      license='GPL',
      packages=['xnn'],
      install_requires=[
          'matplotlib>=2.2.2','tensorflow>=2.0.0b0', 'numpy>=1.15.2', 'interpret'],
      zip_safe=False)
