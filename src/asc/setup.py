from glob import glob

from setuptools import find_packages, setup

package_name = 'asc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lisa',
    maintainer_email='lisa.yamamoto@student.unimelb.edu.au',
    description='Deploy dual-architecture VLAs on ASC robot and in sim',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_capture = asc.camera_capture:main',
            'odometry = asc.odometry:main',
            'outer_loop_controller = asc.outer_loop_controller:main',
            'inner_loop_controller = asc.inner_loop_controller:main',
            'roboclaw_for_motors = asc.roboclaw_for_motors:main',
        ],
    },
)
