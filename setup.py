from setuptools import setup, find_packages

setup(
    name = 'pensmodule',
    version = '0.10',
    author = 'ICT&Microsoft',
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
       'pandas>=1.1.5',
       'rouge>=0.3.1',
       'tensorboardX>=1.9'
    ]
)



