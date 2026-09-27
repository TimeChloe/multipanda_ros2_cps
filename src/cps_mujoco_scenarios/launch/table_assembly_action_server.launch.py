from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cps_mujoco_scenarios',
            executable='table_assembly_action_server',
            name='table_assembly_action_server',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
    ])
