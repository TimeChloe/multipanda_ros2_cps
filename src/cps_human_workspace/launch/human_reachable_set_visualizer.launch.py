from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
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
        Node(
            package='cps_human_workspace',
            executable='human_reachable_set_visualizer',
            name='human_reachable_set_visualizer',
            output='screen',
            parameters=[{
                'reachable_set_topic': LaunchConfiguration(
                    'reachable_set_topic'),
                'marker_topic': LaunchConfiguration('marker_topic'),
                'marker_alpha': ParameterValue(
                    LaunchConfiguration('marker_alpha'), value_type=float),
                'marker_lifetime_sec': ParameterValue(
                    LaunchConfiguration('marker_lifetime_sec'),
                    value_type=float),
            }],
        ),
    ])
