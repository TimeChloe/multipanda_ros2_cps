from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from math import pi
import os

import xacro
def concatenate_ns(ns1, ns2, absolute=False):
    
    if(len(ns1) == 0):
        return ns2
    if(len(ns2) == 0):
        return ns1
    
    # check for /s at the end and start
    if(ns1[0] == '/'):
        ns1 = ns1[1:]
    if(ns1[-1] == '/'):
        ns1 = ns1[:-1]
    if(ns2[0] == '/'):
        ns2 = ns2[1:]
    if(ns2[-1] == '/'):
        ns2 = ns2[:-1]
    if(absolute):
        ns1 = '/' + ns1
    return ns1 + '/' + ns2

def generate_launch_description():

    # Fixed variables
    franka_xacro_file = os.path.join(get_package_share_directory('franka_description'), 'robots', 'sim',
                                     'panda_arm_sim.urdf.xacro')
    xml_file = os.path.join(get_package_share_directory('franka_description'), 'mujoco', 'franka', 'scene.xml')
    mjros_config_file = os.path.join(get_package_share_directory('franka_bringup'), 'config', 'sim',
                                     'single_sim_controllers.yaml')
    franka_bringup_path = get_package_share_directory('franka_bringup')

    # Parameters as launch arguments
    load_gripper_param = 'load_gripper'
    arm_id_param = 'arm_id'
    initial_positions_param = 'initial_positions'
    
    load_gripper = LaunchConfiguration(load_gripper_param)
    arm_id = LaunchConfiguration(arm_id_param)
    initial_positions = LaunchConfiguration(initial_positions_param)

    robot_description = Command(
        [FindExecutable(name='xacro'), ' ', franka_xacro_file, 
            ' arm_id:=', arm_id, 
            ' hand:=', load_gripper,
            ' initial_positions:=', initial_positions])
    
    params = {'robot_description': robot_description}
    ns = ''     # this must match the namespace argument under mujoco_ros2_control in the plugin's parameter yaml file. 
                # See the ros2_control_plugins_example_with_ns.yaml file for more details.

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        namespace= ns,
        parameters=[params]
    )
    node_joint_state_broadcaster = Node( # RVIZ dependency
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            namespace= ns,
            parameters=[
                {'source_list': [concatenate_ns(ns, 'joint_states', True)],
                 'rate': 10}],
    )
    return LaunchDescription([
        # Launch args
        DeclareLaunchArgument(
            load_gripper_param,
            default_value='true',
            description='Use Franka Gripper as an end-effector, otherwise, the robot is loaded '
                        'without an end-effector. Defaults to true.'),
        DeclareLaunchArgument(
            arm_id_param,
            default_value='panda',
            description='The name of the robot. Defaults to panda.'),
        DeclareLaunchArgument(
            initial_positions_param,
            default_value='"0.0 -0.785 0.0 -2.356 0.0 1.571 0.785"',
            description='Initial joint positions of the robot. Must be enclosed in quotes, and in pure number.'
                        'Defaults to the "communication_test" pose.'),

        # Mujoco ros2 server launch
        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(franka_bringup_path + '/launch/sim/launch_mujoco_ros_server.launch'),
            launch_arguments={
                'use_sim_time': "true",
                'modelfile': xml_file,
                'verbose': "true",
                'ns': ns,
                'mujoco_plugin_config': mjros_config_file
                # 'mujoco_plugin_config': os.path.join(mjr2_control_path, 'example', 'ros2_control_plugins_example.yaml')

            }.items()
        ),

        # Miscellaneous
        node_robot_state_publisher,
        node_joint_state_broadcaster,

        Node( # RVIZ dependency; broken right now
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '-c', concatenate_ns(ns, 'controller_manager', True)],
            output='screen',
        ),
    ])
