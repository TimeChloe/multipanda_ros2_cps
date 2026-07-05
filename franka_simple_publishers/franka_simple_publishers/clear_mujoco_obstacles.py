import argparse
import sys

import rclpy
from mujoco_ros_msgs.srv import SetGeomProperties


DEFAULT_GEOMS = (
    'table_top',
    'table_leg_fl',
    'table_leg_fr',
    'table_leg_rl',
    'table_leg_rr',
    'spring_seg_01',
    'spring_seg_02',
    'spring_seg_03',
    'spring_seg_04',
    'spring_seg_05',
    'spring_seg_06',
    'spring_seg_07',
    'spring_seg_08',
    'spring_seg_09',
    'spring_seg_10',
    'hand_surface_pad',
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Shrink the table, spring, and surface geoms in MuJoCo.')
    parser.add_argument(
        '--service',
        default='/set_geom_properties',
        help='MuJoCo SetGeomProperties service name.')
    parser.add_argument(
        '--timeout',
        type=float,
        default=5.0,
        help='Seconds to wait for the service and each response.')
    parser.add_argument(
        '--size',
        type=float,
        default=1.0e-6,
        help='Replacement geom size in meters.')
    parser.add_argument(
        '--admin-hash',
        default='',
        help='Admin hash for MuJoCo eval mode, if required.')
    parser.add_argument(
        '--geom',
        action='append',
        dest='geoms',
        help=(
            'Geom name to clear. May be repeated. Defaults clear the table '
            'scenario.'))
    return parser.parse_args(argv)


def make_request(geom_name, size, admin_hash):
    request = SetGeomProperties.Request()
    request.properties.name = geom_name
    request.set_collision = True
    request.properties.contype = 0
    request.properties.conaffinity = 0
    request.set_size = True
    request.properties.size_0 = size
    request.properties.size_1 = size
    request.properties.size_2 = size
    request.set_friction = True
    request.properties.friction_slide = 0.0
    request.properties.friction_spin = 0.0
    request.properties.friction_roll = 0.0
    request.admin_hash = admin_hash
    return request


def clear_geom(node, client, geom_name, size, admin_hash, timeout):
    future = client.call_async(make_request(geom_name, size, admin_hash))
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)

    if not future.done():
        return False, 'timed out'

    response = future.result()
    if response is None:
        return False, 'service returned no response'

    return response.success, response.status_message


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    geoms = args.geoms if args.geoms else DEFAULT_GEOMS
    size = max(args.size, 1.0e-9)

    rclpy.init()
    node = rclpy.create_node('clear_mujoco_obstacles')
    client = node.create_client(SetGeomProperties, args.service)

    try:
        if not client.wait_for_service(timeout_sec=args.timeout):
            node.get_logger().error(
                "Service '%s' is not available." % args.service)
            return 1

        failures = []
        for geom_name in geoms:
            success, message = clear_geom(
                node,
                client,
                geom_name,
                size,
                args.admin_hash,
                args.timeout)
            if success:
                node.get_logger().info("Cleared '%s'." % geom_name)
            else:
                failures.append((geom_name, message))
                node.get_logger().warn(
                    "Could not clear '%s': %s" % (geom_name, message))

        if failures:
            node.get_logger().warn(
                'Finished with %d failed geom updates.' % len(failures))
            return 1

        node.get_logger().info('Obstacle clear command completed.')
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
