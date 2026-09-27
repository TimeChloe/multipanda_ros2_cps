from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    human_workspace_package = LaunchConfiguration('human_workspace_package')
    human_workspace_config = LaunchConfiguration('human_workspace_config')

    default_config_path = PathJoinSubstitution([
        FindPackageShare(human_workspace_package),
        'config',
        human_workspace_config,
    ])
    reachable_set_visualizer_launch = PathJoinSubstitution([
        FindPackageShare('cps_human_workspace'),
        'launch',
        'human_reachable_set_visualizer.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'human_workspace_package',
            default_value='cps_human_workspace',
            description='ROS package that owns the human workspace config.'),
        DeclareLaunchArgument(
            'human_workspace_config',
            default_value='human_workspace.yaml',
            description='Config file under <human_workspace_package>/config.'),
        DeclareLaunchArgument(
            'human_workspace_config_path',
            default_value=default_config_path,
            description='Absolute config path. Overrides package/config when set.'),
        DeclareLaunchArgument(
            'frame_id',
            default_value='panda_link0',
            description='Frame of the published human observation.'),
        DeclareLaunchArgument(
            'state_topic',
            default_value='human_workspace/state',
            description='HumanWorkspace state topic consumed by controllers.'),
        DeclareLaunchArgument(
            'publish_rate',
            default_value=str(50.0),
            description='Hand-observation publication rate in Hz (default: 30 ms period).'),
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
            description='Use simulated time.'),
        Node(
            package='cps_human_workspace',
            executable='human_workspace_publisher',
            name='human_workspace_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'), value_type=bool),
                'human_workspace_config_path': LaunchConfiguration(
                    'human_workspace_config_path'),
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
