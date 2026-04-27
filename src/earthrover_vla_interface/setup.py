from setuptools import find_packages, setup

package_name = 'earthrover_vla_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lisa',
    maintainer_email='164627499+lyamatomato@users.noreply.github.com',
    description='Bridge between VLA and robot in sim or hardware',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vla_bridge_node = earthrover_vla_interface.vla_bridge_node:main',
        ],
    },
)
