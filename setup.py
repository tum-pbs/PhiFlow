from os.path import join, dirname

from setuptools import setup

try:
    with open(join(dirname(__file__), 'docs/Package_Info.md'), 'r') as readme:
        long_description = readme.read()
except FileNotFoundError:
    long_description = ""
    pass

with open(join(dirname(__file__), 'phi', 'VERSION'), 'r') as version_file:
    version = version_file.read()

setup(
    name='phiflow',
    version=version,
    download_url='https://github.com/tum-pbs/PhiFlow/archive/%s.tar.gz' % version,
    packages=['phi',
              'phi.field',
              'phi.geom',
              'phi.jax',
              'phi.jax.stax',
              'phi.math',
              'phi.physics',
              'phi.tf',
              'phi.torch',
              'phi.vis',
              'phi.vis._console',
              'phi.vis._dash',
              'phi.vis._matplotlib',
          ],
    description='Differentiable PDE solving framework for machine learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['Differentiable', 'Simulation', 'Fluid', 'Machine Learning', 'Deep Learning'],
    license='MIT',
    author='Philipp Holl',
    author_email='philipp.holl@tum.de',
    url='https://github.com/tum-pbs/PhiFlow',
    include_package_data=True,
    install_requires=[
        'phiml',
        'matplotlib>=3.5.0',  # also required by dash for color maps
        'packaging',
    ],
    # Optional packages:  dash + plotly (included in dash)
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
