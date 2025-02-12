from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
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
    garmi_xacro_file = os.path.join(get_package_share_directory('garmi_description'), 'robots',
                                     'garmi_sim.urdf.xacro')
    default_scene_xml_file = os.path.join(get_package_share_directory('garmi_description'), 'mujoco', 'garmi', 'assets', 'xml', 'garmi.xml')
    mjros_config_file = os.path.join(get_package_share_directory('garmi_bringup'), 'config', 'sim',
                                     'sim_garmi.yaml')

    xml_path = default_scene_xml_file
    xacro_file = garmi_xacro_file
    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    params = {'robot_description': doc.toxml()}
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
        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(get_package_share_directory('garmi_bringup') + '/launch/sim/launch_mujoco_ros_server.launch'),
            launch_arguments={
                'use_sim_time': "true",
                'modelfile': xml_path,
                'verbose': "true",
                'ns': ns,
                'mujoco_plugin_config': mjros_config_file

            }.items()
        ),
        node_robot_state_publisher,
        # Node( # RVIZ dependency; broken right now
        #     package='controller_manager',
        #     executable='spawner',
        #     arguments=['joint_state_broadcaster', '-c', concatenate_ns(ns, 'controller_manager', True)],
        #     output='screen',
        # ),
        node_joint_state_broadcaster
    ])
