from glob import glob

from setuptools import find_packages, setup

package_name = 'async_vla'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lisa',
    maintainer_email='lisa.yamamoto@student.unimelb.edu.au',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sys2 = async_vla.sys2:main',
            'sys1 = async_vla.sys1:main',
        ],
    },
)
