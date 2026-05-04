from setuptools import setup

package_name = 'frodo_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/frodo_sdk_bridge_launch.py']),
    ],
    install_requires=['requests', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='miahv',
    maintainer_email='miahv@todo.todo',
    description='Frodo SDK bridge package for ROS2.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'frodo_sdk_bridge = frodo_bringup.frodo_sdk_bridge:main',
        ],
    },
)
