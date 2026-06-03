from glob import glob

from setuptools import find_packages, setup

package_name = 'asclinic_vla'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.py')),
        (f'share/{package_name}/config', glob('config/*')),
        (f'share/{package_name}/docs', glob('docs/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='miahv',
    maintainer_email='miahv@example.com',
    description='ASClinic robot integration package for AsyncVLA trajectory-driven navigation.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'goal_publisher = asclinic_vla.goal_publisher_node:main',
            'asclinic_camera_capture = asclinic_vla.camera_capture_node:main',
            'vla_http_client = asclinic_vla.vla_http_client_node:main',
            'action_chunk_pure_pursuit = asclinic_vla.action_chunk_pure_pursuit_node:main',
            'encoder_odometry = asclinic_vla.encoder_odometry_node:main',
            'odom_action_chunk_tracker = asclinic_vla.odom_action_chunk_tracker_node:main',
            'fake_action_chunk_publisher = asclinic_vla.fake_action_chunk_publisher_node:main',
            'odom_to_pose2d = asclinic_vla.odom_to_pose2d_node:main',
            'wheel_reference_to_cmd_vel = asclinic_vla.wheel_reference_to_cmd_vel_node:main',
            'zenoh_camera_bridge = asclinic_vla.zenoh_camera_bridge_node:main',
            'zenoh_instruction_bridge = asclinic_vla.zenoh_instruction_bridge_node:main',
            'zenoh_cmd_vel_bridge = asclinic_vla.zenoh_cmd_vel_bridge_node:main',
            'zenoh_action_chunk_bridge = asclinic_vla.zenoh_action_chunk_bridge_node:main',
            'split_action_head = asclinic_vla.split_action_head_runner:main',
            'benchmark_action_head = asclinic_vla.benchmark_action_head:main',
            'vla_performance_sim = asclinic_vla.vla_performance_sim_node:main',
            'vla_performance_hardware = asclinic_vla.vla_performance_hardware_node:main',
            'wheel_pid_controller = asclinic_vla.wheel_pid_controller_node:main',
            'roboclaw_for_motors = asclinic_vla.roboclaw_for_motors_node:main',
        ],
    },
)
