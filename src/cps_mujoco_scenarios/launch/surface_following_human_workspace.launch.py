from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('cps_mujoco_scenarios'),
        'config',
        'human_workspace_surface_following.yaml',
    ])
    reachable_set_visualizer_launch = PathJoinSubstitution([
        FindPackageShare('cps_human_workspace'),
        'launch',
        'human_reachable_set_visualizer.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'human_workspace_config_path',
            default_value=default_config,
            description='SaRA BodyPartCombined limits for the surface-following hand.'),
        DeclareLaunchArgument(
            'surface_body_name',
            default_value='human_hand_surface',
            description='MuJoCo body whose center is used as the measured hand center.'),
        DeclareLaunchArgument(
            'body_state_service',
            default_value='/get_body_state',
            description='MuJoCo GetBodyState service.'),
        DeclareLaunchArgument(
            'frame_id',
            default_value='panda_link0',
            description=(
                'MuJoCo body frame used as the output frame. The node reads its '
                'world pose and transforms surface observations into this frame.')),
        DeclareLaunchArgument(
            'state_topic',
            default_value='human_workspace/state',
            description='Workspace observation topic consumed by the controller.'),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='50.0',
            description='MuJoCo surface sampling rate in Hz.'),
        DeclareLaunchArgument(
            'reachable_set_topic',
            default_value='/human_workspace/reachable_set',
            description='Calculated human reachable-set input topic.'),
        DeclareLaunchArgument(
            'marker_topic',
            default_value='/human_workspace/markers',
            description='RViz MarkerArray output topic.'),
        DeclareLaunchArgument(
            'marker_alpha',
            default_value='0.3',
            description='Blue reachable-sphere opacity.'),
        DeclareLaunchArgument(
            'marker_lifetime_sec',
            default_value='1.0',
            description='Marker lifetime after the last reachable-set update.'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use MuJoCo simulated time for message stamps.'),
        Node(
            package='cps_mujoco_scenarios',
            executable='surface_following_human_workspace',
            name='surface_following_human_workspace',
            output='screen',
            parameters=[{
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'), value_type=bool),
                'human_workspace_config_path': LaunchConfiguration(
                    'human_workspace_config_path'),
                'surface_body_name': LaunchConfiguration('surface_body_name'),
                'body_state_service': LaunchConfiguration('body_state_service'),
                'frame_id': LaunchConfiguration('frame_id'),
                'state_topic': LaunchConfiguration('state_topic'),
                'publish_rate': ParameterValue(
                    LaunchConfiguration('publish_rate'), value_type=float),
            }],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(reachable_set_visualizer_launch),
            launch_arguments={
                'reachable_set_topic': LaunchConfiguration(
                    'reachable_set_topic'),
                'marker_topic': LaunchConfiguration('marker_topic'),
                'marker_alpha': LaunchConfiguration('marker_alpha'),
                'marker_lifetime_sec': LaunchConfiguration(
                    'marker_lifetime_sec'),
            }.items(),
        ),
    ])
