from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _float_launch_config(context, name):
    return float(LaunchConfiguration(name).perform(context))


def _launch_setup(context, *args, **kwargs):
    del args, kwargs

    human_workspace_config_path = LaunchConfiguration('human_workspace_config_path')
    frame_id = LaunchConfiguration('frame_id')
    marker_topic = LaunchConfiguration('marker_topic')
    state_topic = LaunchConfiguration('state_topic')
    publish_rate = LaunchConfiguration('publish_rate')
    marker_lifetime_sec = LaunchConfiguration('marker_lifetime_sec')
    use_sim_time = LaunchConfiguration('use_sim_time')
    visualize_ee_collision_area = LaunchConfiguration('visualize_ee_collision_area')
    ee_frame_id = LaunchConfiguration('ee_frame_id')
    ee_collision_radius = LaunchConfiguration('ee_collision_radius')
    tracking_pos_error_bound = LaunchConfiguration('tracking_pos_error_bound')
    ee_collision_center_offset = [
        _float_launch_config(context, 'ee_collision_center_offset_x'),
        _float_launch_config(context, 'ee_collision_center_offset_y'),
        _float_launch_config(context, 'ee_collision_center_offset_z'),
    ]

    return [Node(
        package='cps_human_workspace',
        executable='human_workspace_visualizer',
        name='human_workspace_visualizer',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            'human_workspace_config_path': human_workspace_config_path,
            'frame_id': frame_id,
            'marker_topic': marker_topic,
            'state_topic': state_topic,
            'publish_rate': ParameterValue(publish_rate, value_type=float),
            'marker_lifetime_sec': ParameterValue(
                marker_lifetime_sec,
                value_type=float),
            'visualize_ee_collision_area': ParameterValue(
                visualize_ee_collision_area,
                value_type=bool),
            'ee_frame_id': ee_frame_id,
            'ee_collision_radius': ParameterValue(
                ee_collision_radius,
                value_type=float),
            'ee_collision_center_offset': ee_collision_center_offset,
            'tracking_pos_error_bound': ParameterValue(
                tracking_pos_error_bound,
                value_type=float),
        }],
    )]


def generate_launch_description():
    human_workspace_package = LaunchConfiguration('human_workspace_package')
    human_workspace_config = LaunchConfiguration('human_workspace_config')

    default_config_path = PathJoinSubstitution([
        FindPackageShare(human_workspace_package),
        'config',
        human_workspace_config,
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
            description='Frame used for human workspace markers.'),
        DeclareLaunchArgument(
            'marker_topic',
            default_value='human_workspace/markers',
            description='MarkerArray topic for RViz visualization.'),
        DeclareLaunchArgument(
            'state_topic',
            default_value='human_workspace/state',
            description='HumanWorkspace state topic consumed by controllers.'),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='10.0',
            description='Marker publication rate in Hz.'),
        DeclareLaunchArgument(
            'marker_lifetime_sec',
            default_value='0.1',
            description='Marker lifetime. Use 0.0 to keep markers forever.'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulated time.'),
        DeclareLaunchArgument(
            'visualize_ee_collision_area',
            default_value='true',
            description='Also visualize the monitored end-effector collision area.'),
        DeclareLaunchArgument(
            'ee_frame_id',
            default_value='panda_metal_ball_link',
            description='Frame at the center of the monitored EE collision model.'),
        DeclareLaunchArgument(
            'ee_collision_radius',
            default_value='0.03',
            description='Radius expanded around the EE collision center in meters.'),
        DeclareLaunchArgument(
            'ee_collision_center_offset_x',
            default_value='0.0',
            description='Collision-center x offset from ee_frame_id in meters.'),
        DeclareLaunchArgument(
            'ee_collision_center_offset_y',
            default_value='0.0',
            description='Collision-center y offset from ee_frame_id in meters.'),
        DeclareLaunchArgument(
            'ee_collision_center_offset_z',
            default_value='0.0',
            description='Collision-center z offset from ee_frame_id in meters.'),
        DeclareLaunchArgument(
            'tracking_pos_error_bound',
            default_value='0.005',
            description='Tracking position error bound in meters.'),
        OpaqueFunction(function=_launch_setup),
    ])
