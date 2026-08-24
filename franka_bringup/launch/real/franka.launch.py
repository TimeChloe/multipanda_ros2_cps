# Copyright (c) 2021 Franka Emika GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from franka_bringup.tool_model import generate_tool_artifacts
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    Shutdown,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _as_bool(value):
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _launch_setup(context):
    robot_ip = LaunchConfiguration('robot_ip')
    robot_ip_value = robot_ip.perform(context)
    arm_id = LaunchConfiguration('arm_id').perform(context)
    load_gripper = LaunchConfiguration('load_gripper')
    load_gripper_value = _as_bool(load_gripper.perform(context))
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    fake_sensor_commands = LaunchConfiguration('fake_sensor_commands')
    use_rviz = LaunchConfiguration('use_rviz')
    tool_config = LaunchConfiguration('tool_config').perform(context)
    output_root = LaunchConfiguration('tool_output_dir').perform(context)

    if tool_config and load_gripper_value:
        raise RuntimeError(
            'tool_config and load_gripper=true cannot be combined in schema version 1; '
            'describe the complete gripper/tool assembly in one tool.yaml instead'
        )

    description_share = get_package_share_directory('franka_description')
    robot_xacro = os.path.join(
        description_share, 'robots', 'real', 'panda_arm.urdf.xacro'
    )
    monitor_base = os.path.join(
        description_share, 'model_urdf', 'panda_ng_monitor_base.urdf'
    )
    controller_config = os.path.join(
        get_package_share_directory('franka_bringup'),
        'config',
        'real',
        'single_controllers.yaml',
    )
    artifacts = generate_tool_artifacts(
        tool_config_path=tool_config,
        arm_id=arm_id,
        robot_xacro_path=robot_xacro,
        xacro_mappings={
            'arm_id': arm_id,
            'hand': 'true' if load_gripper_value else 'false',
            'robot_ip': robot_ip_value,
            'use_fake_hardware': use_fake_hardware.perform(context),
            'fake_sensor_commands': fake_sensor_commands.perform(context),
            'metal_ball': 'false',
        },
        monitor_base_urdf_path=monitor_base,
        output_root=output_root,
        controller_config_path=controller_config,
    )

    rviz_file = os.path.join(
        description_share, 'rviz', 'visualize_franka.rviz'
    )
    actions = [
        LogInfo(
            msg=(
                f"Unified real-robot tool model: "
                f"{artifacts.tool.name if artifacts.tool else 'none'}; "
                "libfranka load policy: strict one-time read-only check"
            )
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': artifacts.robot_description}],
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[{
                'source_list': ['franka/joint_states', 'panda_gripper/joint_states'],
                'rate': 30,
            }],
        ),
        Node(
            package='franka_control2',
            executable='franka_control2_node',
            parameters=[
                {'robot_description': artifacts.robot_description},
                artifacts.controller_config_path,
            ],
            remappings=[('joint_states', 'franka/joint_states')],
            output={'stdout': 'screen', 'stderr': 'screen'},
            on_exit=Shutdown(),
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['franka_robot_state_broadcaster'],
            output='screen',
            condition=UnlessCondition(use_fake_hardware),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('franka_gripper'), 'launch', 'gripper.launch.py'
                ])
            ]),
            launch_arguments={
                'robot_ip': robot_ip,
                'use_fake_hardware': use_fake_hardware,
            }.items(),
            condition=IfCondition(load_gripper),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['--display-config', rviz_file],
            condition=IfCondition(use_rviz),
        ),
    ]
    if artifacts.tool is not None:
        actions.append(
            Node(
                package='franka_bringup',
                executable='tool_load_validator',
                name='tool_load_validator',
                output='screen',
                parameters=[{
                    'tool_config': tool_config,
                    'state_topic': '/franka_robot_state_broadcaster/robot_state',
                }],
                condition=UnlessCondition(use_fake_hardware),
                on_exit=Shutdown(
                    reason=(
                        'Tool-load validator exited; stopping the real-robot launch.'
                    )
                ),
            )
        )
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', description='Hostname or IP address of the robot.'),
        DeclareLaunchArgument(
            'arm_id', default_value='panda', description='Robot arm identifier.'
        ),
        DeclareLaunchArgument(
            'use_rviz', default_value='false', description='Visualize the robot in Rviz.'
        ),
        DeclareLaunchArgument(
            'use_fake_hardware', default_value='false', description='Use fake hardware.'
        ),
        DeclareLaunchArgument(
            'fake_sensor_commands',
            default_value='false',
            description='Fake sensor commands; only valid with fake hardware.',
        ),
        DeclareLaunchArgument(
            'load_gripper',
            default_value='false',
            description=(
                'Load the standard Franka gripper when no unified tool_config '
                'is supplied.'
            ),
        ),
        DeclareLaunchArgument(
            'tool_config',
            default_value='',
            description='Optional unified tool.yaml for URDF and libfranka load validation.',
        ),
        DeclareLaunchArgument(
            'tool_output_dir',
            default_value='/tmp/multipanda_tool_models',
            description='Directory for generated launch-time robot artifacts.',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
