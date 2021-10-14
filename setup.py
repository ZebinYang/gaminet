from setuptools import setup

package_data = {
    "gaminet": [
        "lib/lib_ebmcore_win_x64.dll",
        "lib/lib_ebmcore_linux_x64.so",
        "lib/lib_ebmcore_mac_x64.dylib",
        "lib/lib_ebmcore_win_x64_debug.dll",
        "lib/lib_ebmcore_linux_x64_debug.so",
        "lib/lib_ebmcore_mac_x64_debug.dylib",
        "lib/lib_ebmcore_win_x64.pdb",
        "lib/lib_ebmcore_win_x64_debug.pdb",
    ]
}

setup(name='gaminet',
      version='0.5.5',
      description='Explainable Neural Networks based on Generalized Additive Models with Structured Interactions',
      url='https://github.com/ZebinYang/GAMINet',
      author='Zebin Yang',
      author_email='yangzb2010@connect.hku.hk',
      license='GPL',
      packages=['gaminet'],
      package_data=package_data,
      install_requires=['matplotlib>=3.1.3', 'tensorflow>=2.0.0', 'numpy>=1.15.2', 'pandas>=0.19.2', 'scikit-learn>=0.23.0', 'tensorflow_lattice>=2.0.8'],
      zip_safe=False)
