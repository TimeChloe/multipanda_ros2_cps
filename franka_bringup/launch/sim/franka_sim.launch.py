# Copyright (c) 2026 Yue
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
)
from launch.conditions import IfCondition
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def concatenate_ns(ns1, ns2, absolute=False):
    if len(ns1) == 0:
        return ns2
    if len(ns2) == 0:
        return ns1
    ns1 = ns1.strip('/')
    ns2 = ns2.strip('/')
    if absolute:
        ns1 = '/' + ns1
    return ns1 + '/' + ns2


def _strip_outer_quotes(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _launch_setup(context):
    arm_id = LaunchConfiguration('arm_id').perform(context)
    initial_positions = _strip_outer_quotes(
        LaunchConfiguration('initial_positions').perform(context)
    )
    scene = LaunchConfiguration('scene').perform(context)
    tool_config = LaunchConfiguration('tool_config').perform(context)
    output_root = LaunchConfiguration('tool_output_dir').perform(context)
    use_rviz = LaunchConfiguration('use_rviz')

    description_share = get_package_share_directory('franka_description')
    bringup_share = get_package_share_directory('franka_bringup')
    robot_xacro = os.path.join(
        description_share, 'robots', 'sim', 'panda_arm_sim.urdf.xacro'
    )
    monitor_base = os.path.join(
        description_share, 'model_urdf', 'panda_ng_monitor_base.urdf'
    )
    mujoco_base = os.path.join(
        description_share, 'mujoco', 'franka', 'panda_ng.xml'
    )
    table_model = os.path.join(
        description_share, 'mujoco', 'franka', 'table.xml'
    )
    controller_config = os.path.join(
        bringup_share, 'config', 'sim', 'single_sim_controllers.yaml'
    )

    artifacts = generate_tool_artifacts(
        tool_config_path=tool_config,
        arm_id=arm_id,
        robot_xacro_path=robot_xacro,
        xacro_mappings={
            'arm_id': arm_id,
            'hand': 'false',
            'metal_ball': 'false',
            'initial_positions': initial_positions,
        },
        monitor_base_urdf_path=monitor_base,
        output_root=output_root,
        mujoco_base_path=mujoco_base,
        mujoco_table_path=table_model,
        include_table=scene == 'table_spring',
        controller_config_path=controller_config,
    )

    ns = ''
    params = {'robot_description': artifacts.robot_description}
    rviz_file = os.path.join(
        description_share, 'rviz', 'visualize_franka.rviz'
    )
    tool_name = artifacts.tool.name if artifacts.tool else 'none'

    return [
        LogInfo(
            msg=(
                f"Unified tool model: {tool_name}; "
                f"monitor URDF: {artifacts.monitor_urdf_path}; "
                f"MuJoCo scene: {artifacts.mujoco_scene_path}"
            )
        ),
        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(
                os.path.join(
                    bringup_share,
                    'launch',
                    'sim',
                    'launch_mujoco_ros_server.launch',
                )
            ),
            launch_arguments={
                'use_sim_time': 'true',
                'modelfile': artifacts.mujoco_scene_path,
                'verbose': 'true',
                'ns': ns,
                'mujoco_plugin_config': artifacts.controller_config_path,
            }.items(),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            namespace=ns,
            parameters=[params],
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            namespace=ns,
            parameters=[{
                'source_list': [concatenate_ns(ns, 'joint_states', True)],
                'rate': 30,
            }],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'joint_state_broadcaster',
                '-c',
                concatenate_ns(ns, 'controller_manager', True),
            ],
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['--display-config', rviz_file],
            condition=IfCondition(use_rviz),
        ),
    ]


def generate_launch_description():
    bringup_share = get_package_share_directory('franka_bringup')
    default_tool = os.path.join(
        bringup_share, 'config', 'tools', 'metal_ball.yaml'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz', default_value='true', description='Visualize the robot in Rviz.'
        ),
        DeclareLaunchArgument(
            'arm_id',
            default_value='panda',
            description='Robot name; single MuJoCo arm requires panda.',
        ),
        DeclareLaunchArgument(
            'initial_positions',
            default_value='"0.0 -0.578 0.0 -1.753 0.0 1.175 0.785"',
            description='Initial joint positions.',
        ),
        DeclareLaunchArgument(
            'scene',
            default_value='no_table',
            choices=['no_table', 'table_spring'],
            description=(
                'MuJoCo scene: no_table loads robot and floor; table_spring adds '
                'the table and compliant hand-surface pad.'
            ),
        ),
        DeclareLaunchArgument(
            'tool_config',
            default_value=default_tool,
            description=(
                'Unified tool.yaml used by MuJoCo, Pinocchio, RViz and '
                'safety geometry.'
            ),
        ),
        DeclareLaunchArgument(
            'tool_output_dir',
            default_value='/tmp/multipanda_tool_models',
            description='Directory for generated launch-time robot artifacts.',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
