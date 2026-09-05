#!/usr/bin/env python3

import argparse
import sys

import rclpy
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from cps_mujoco_scenarios.action import MoveTableAssembly
from mujoco_ros_msgs.msg import ScalarStamped
from mujoco_ros_msgs.srv import SetPause
from panda_motion_generator_msgs.action import CartesianViaMotion


DEFAULT_VIA_POSES = (
    (0.306957, 0.0, 0.779912, 0.923956, -0.382499, 0.0, 0.0),
    (0.306957, 0.0, 0.619912, 0.923956, -0.382499, 0.0, 0.0),
    (0.306957, 0.0, 0.449912, 0.923956, -0.382499, 0.0, 0.0),
    (0.306957, 0.0, 0.279912, 0.923956, -0.382499, 0.0, 0.0),
    (0.306957, 0.0, 0.179912, 0.923956, -0.382499, 0.0, 0.0),
    (0.306957, 0.0, 0.079912, 0.923956, -0.382499, 0.0, 0.0),
    (0.306957, 0.0, 0.029912, 0.923956, -0.382499, 0.0, 0.0),
)


class StaticDownCollisionDisappearTest(Node):
    def __init__(self, args):
        super().__init__("static_down_collision_disappear_test")
        self.args = args
        self.contact_seen = False
        self.contact_time = None
        self.disappear_started = False
        self.exit_code = 0
        self.test_started_time = None

        self.contact_sub = self.create_subscription(
            ScalarStamped,
            args.contact_topic,
            self.handle_contact,
            10)

        self.action_client = ActionClient(
            self, CartesianViaMotion, args.action_name)
        self.scene_action_client = ActionClient(
            self, MoveTableAssembly, args.scene_action_name)
        self.pause_client = self.create_client(SetPause, args.pause_service)

        self.start_timer = self.create_timer(0.1, self.try_start)

    def try_start(self):
        if not self.action_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().info(
                f"Waiting for action server {self.args.action_name}...",
                throttle_duration_sec=2.0)
            return

        self.start_timer.cancel()
        self.unpause_then_send_goal()

    def unpause_then_send_goal(self):
        if not self.args.unpause:
            self.send_goal()
            return

        if not self.pause_client.wait_for_service(timeout_sec=self.args.service_timeout):
            self.get_logger().warn(
                f"Pause service {self.args.pause_service} is not available; "
                "continuing without an automatic unpause.")
            self.send_goal()
            return

        request = SetPause.Request()
        request.paused = False
        request.admin_hash = self.args.admin_hash

        self.get_logger().info(f"Unpausing MuJoCo through {self.args.pause_service}.")
        future = self.pause_client.call_async(request)
        future.add_done_callback(self.handle_unpause_result)

    def handle_unpause_result(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Unpause service call failed: {exc}")
            self.send_goal()
            return

        if not response.success:
            self.get_logger().warn("Unpause service returned success=false.")
        else:
            self.get_logger().info("MuJoCo is unpaused.")
        self.send_goal()

    def send_goal(self):
        goal = CartesianViaMotion.Goal()
        goal.via_poses = [self.make_pose(values) for values in DEFAULT_VIA_POSES]

        self.get_logger().info(
            f"Sending straight-down goal with {len(goal.via_poses)} poses.")
        self.test_started_time = self.get_clock().now()
        self.contact_timeout_timer = self.create_timer(0.5, self.check_contact_timeout)
        future = self.action_client.send_goal_async(
            goal, feedback_callback=self.handle_feedback)
        future.add_done_callback(self.handle_goal_response)

    @staticmethod
    def make_pose(values):
        x, y, z, qx, qy, qz, qw = values
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    def handle_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to send straight-down goal: {exc}")
            self.exit_code = 2
            rclpy.shutdown()
            return

        if not goal_handle.accepted:
            self.get_logger().error("Straight-down goal was rejected.")
            self.exit_code = 2
            rclpy.shutdown()
            return

        self.get_logger().info("Straight-down goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.handle_goal_result)

    def handle_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            "Motion feedback: "
            f"{100.0 * feedback.progress:.1f}% complete, "
            f"{feedback.time_to_completion:.2f}s remaining",
            throttle_duration_sec=1.0)

    def handle_goal_result(self, future):
        try:
            result = future.result().result.result
        except Exception as exc:
            self.get_logger().warn(f"Could not read action result: {exc}")
            return

        if result.state == 0:
            self.get_logger().info("Straight-down goal completed.")
            return

        self.get_logger().warn(
            f"Straight-down goal finished with state {result.state}: {result.error}")

    def handle_contact(self, msg):
        if self.contact_seen or msg.value <= self.args.contact_threshold:
            return

        self.contact_seen = True
        self.contact_time = self.get_clock().now()
        self.contact_timeout_timer.cancel()
        self.get_logger().info(
            f"Collision detected on {self.args.contact_topic}: {msg.value:.6g}. "
            f"Waiting {self.args.disappear_delay:.1f}s before lowering the scene.")
        self.disappear_timer = self.create_timer(0.05, self.maybe_hide_scene)

    def check_contact_timeout(self):
        if self.contact_seen or self.test_started_time is None:
            return

        elapsed = self.get_clock().now() - self.test_started_time
        if elapsed.nanoseconds < int(self.args.contact_timeout * 1.0e9):
            return

        self.get_logger().error(
            f"No contact detected on {self.args.contact_topic} within "
            f"{self.args.contact_timeout:.1f}s.")
        self.exit_code = 5
        rclpy.shutdown()

    def maybe_hide_scene(self):
        if self.disappear_started or self.contact_time is None:
            return

        elapsed = self.get_clock().now() - self.contact_time
        if elapsed.nanoseconds < int(self.args.disappear_delay * 1.0e9):
            return

        self.disappear_started = True
        self.disappear_timer.cancel()
        self.start_hide_scene()

    def start_hide_scene(self):
        if not self.scene_action_client.wait_for_server(
                timeout_sec=self.args.service_timeout):
            self.get_logger().error(
                f"Scene action {self.args.scene_action_name} is not available.")
            self.exit_code = 3
            rclpy.shutdown()
            return

        goal = MoveTableAssembly.Goal()
        goal.target_z_offset = self.args.hide_z_offset
        goal.duration = self.args.hide_duration
        self.get_logger().info(
            "Moving the complete table/spring/surface assembly to "
            f"z offset {goal.target_z_offset:.3f} m over {goal.duration:.3f} s.")
        future = self.scene_action_client.send_goal_async(goal)
        future.add_done_callback(self.handle_hide_goal_response)

    def handle_hide_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(
                f"Could not send table hide action: {exc}")
            self.exit_code = 4
            rclpy.shutdown()
            return

        if not goal_handle.accepted:
            self.get_logger().error("Table hide action was rejected.")
            self.exit_code = 4
            rclpy.shutdown()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.handle_hide_result)

    def handle_hide_result(self, future):
        try:
            result = future.result().result
        except Exception as exc:
            self.get_logger().error(f"Could not read table hide result: {exc}")
            self.exit_code = 4
            rclpy.shutdown()
            return

        if not result.success:
            self.get_logger().error(f"Table hide action failed: {result.message}")
            self.exit_code = 4
        else:
            self.get_logger().info(
                "Table, spring, and hand surface moved below the floor. "
                "Send target_z_offset=0.0 later to raise them again.")
        rclpy.shutdown()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Send a z-axis straight Cartesian test, wait for MuJoCo touch "
            "contact, then move the table/spring/surface below the floor. "
            "Launch cps_human_workspace separately for the static workspace."))
    parser.add_argument(
        "--action-name",
        default="/reachable_cartesian_impedance_controller/follow_cartesian_via_points")
    parser.add_argument("--contact-topic", default="/panda_metal_ball_touch")
    parser.add_argument("--contact-threshold", type=float, default=1.0e-6)
    parser.add_argument("--contact-timeout", type=float, default=60.0)
    parser.add_argument("--disappear-delay", type=float, default=3.0)
    parser.add_argument("--scene-action-name", default="/move_table_assembly")
    parser.add_argument("--hide-z-offset", type=float, default=-0.5)
    parser.add_argument("--hide-duration", type=float, default=3.0)
    parser.add_argument("--pause-service", default="/set_pause")
    parser.add_argument(
        "--no-unpause",
        action="store_false",
        dest="unpause",
        help="Do not call the MuJoCo set_pause service before sending the goal.")
    parser.add_argument("--service-timeout", type=float, default=5.0)
    parser.add_argument("--admin-hash", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.contact_timeout = max(args.contact_timeout, 0.1)
    args.hide_duration = max(args.hide_duration, 0.01)

    rclpy.init()
    node = StaticDownCollisionDisappearTest(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted.")
        node.exit_code = 130
    except ExternalShutdownException:
        pass
    finally:
        exit_code = node.exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
