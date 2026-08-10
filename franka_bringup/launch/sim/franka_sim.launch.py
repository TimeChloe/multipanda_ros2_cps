import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node


def concatenate_ns(ns1, ns2, absolute=False):
    if len(ns1) == 0:
        return ns2
    if len(ns2) == 0:
        return ns1

    if ns1[0] == '/':
        ns1 = ns1[1:]
    if ns1[-1] == '/':
        ns1 = ns1[:-1]
    if ns2[0] == '/':
        ns2 = ns2[1:]
    if ns2[-1] == '/':
        ns2 = ns2[:-1]
    if absolute:
        ns1 = '/' + ns1
    return ns1 + '/' + ns2


def generate_launch_description():
    arm_id_param = 'arm_id'
    initial_positions_param = 'initial_positions'
    use_rviz_param = 'use_rviz'

    arm_id = LaunchConfiguration(arm_id_param)
    initial_positions = LaunchConfiguration(initial_positions_param)
    use_rviz = LaunchConfiguration(use_rviz_param)

    load_gripper = False

    if load_gripper:
        scene_file = 'scene.xml'
    else:
        # scene_file = 'scene_ng.xml'
        scene_file = 'scene_ng_no_table.xml'

    franka_description_path = get_package_share_directory('franka_description')
    franka_bringup_path = get_package_share_directory('franka_bringup')
    franka_xacro_file = os.path.join(
        franka_description_path,
        'robots',
        'sim',
        'panda_arm_sim.urdf.xacro')
    xml_file = os.path.join(
        franka_description_path,
        'mujoco',
        'franka',
        scene_file)
    mjros_config_file = os.path.join(
        franka_bringup_path,
        'config',
        'sim',
        'single_sim_controllers.yaml')
    ns = ''

    robot_description = Command(
        [FindExecutable(name='xacro'), ' ', franka_xacro_file,
         ' arm_id:=', arm_id,
         ' hand:=', str(load_gripper).lower(),
         ' metal_ball:=true',
         ' metal_ball_radius:=0.03',
         ' metal_ball_mass:=0.2',
         ' metal_ball_xyz:="0 0 0.03"',
         ' metal_ball_tcp_xyz:="0 0 0.06"',
         ' initial_positions:=', initial_positions])

    params = {'robot_description': robot_description}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        namespace=ns,
        parameters=[params],
    )

    jsp_source_list = [concatenate_ns(ns, 'joint_states', True)]
    if load_gripper:
        jsp_source_list.append(
            concatenate_ns(ns, 'panda_gripper_sim_node/joint_states', True))

    node_joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=ns,
        parameters=[{
            'source_list': jsp_source_list,
            'rate': 30,
        }],
    )

    rviz_file = os.path.join(
        franka_description_path,
        'rviz',
        'visualize_franka.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            use_rviz_param,
            default_value='true',
            description='Visualize the robot in Rviz.'),
        DeclareLaunchArgument(
            arm_id_param,
            default_value='panda',
            description='The name of the robot. Defaults to panda.'),
        DeclareLaunchArgument(
            initial_positions_param,
            default_value='"0.0 -0.578 0.0 -1.753 0.0 1.175 0.785"',
            description='Initial joint positions of the robot.'),
        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(
                franka_bringup_path +
                '/launch/sim/launch_mujoco_ros_server.launch'),
            launch_arguments={
                'use_sim_time': 'true',
                'modelfile': xml_file,
                'verbose': 'true',
                'ns': ns,
                'mujoco_plugin_config': mjros_config_file,
            }.items(),
        ),
        node_robot_state_publisher,
        node_joint_state_publisher,
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
    ])
