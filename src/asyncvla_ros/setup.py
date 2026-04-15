from setuptools import setup

package_name = 'asyncvla_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='miahv',
    maintainer_email='miahv@example.com',
    description='AsyncVLA ROS2 interface',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'asyncvla_inference_node = asyncvla_ros.asyncvla_inference_node:main',
            'asyncvla_controller_node = asyncvla_ros.asyncvla_controller_node:main',
            'goal_publisher_node = asyncvla_ros.goal_publisher_node:main',
        ],
    },
)