from setuptools import setup, find_namespace_packages

setup(
    name="ai4ar",
    version="0.0.4",
    python_requires='>=3',
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src')
)
