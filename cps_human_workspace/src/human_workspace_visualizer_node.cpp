#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "cps_human_workspace/human_workspace.hpp"

namespace {

using cps_human_workspace::HumanWorkspace;
using Vector3d = Eigen::Vector3d;
using Quaterniond = Eigen::Quaterniond;
using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;

std_msgs::msg::ColorRGBA makeColor(float r, float g, float b, float a) {
  std_msgs::msg::ColorRGBA color;
  color.r = r;
  color.g = g;
  color.b = b;
  color.a = a;
  return color;
}

geometry_msgs::msg::Point toPoint(const Vector3d& p) {
  geometry_msgs::msg::Point point;
  point.x = p.x();
  point.y = p.y();
  point.z = p.z();
  return point;
}

class HumanWorkspaceVisualizer : public rclcpp::Node {
 public:
  HumanWorkspaceVisualizer() : Node("human_workspace_visualizer") {
    start_time_ = now();
    const std::string config_path =
        declare_parameter<std::string>("human_workspace_config_path", "");
    frame_id_ = declare_parameter<std::string>("frame_id", "panda_link0");
    topic_name_ = declare_parameter<std::string>("marker_topic", "human_workspace/markers");
    publish_rate_hz_ = std::max(0.1, declare_parameter<double>("publish_rate", 10.0));
    visualize_ee_collision_area_ =
        declare_parameter<bool>("visualize_ee_collision_area", true);
    ee_frame_id_ = declare_parameter<std::string>("ee_frame_id", "panda_link8");
    ee_collision_radius_ = std::max(
        0.0,
        declare_parameter<double>("ee_collision_radius", 0.04));
    const auto ee_collision_center_offset =
        declare_parameter<std::vector<double>>(
            "ee_collision_center_offset", std::vector<double>{0.0, 0.0, 0.0});
    if (ee_collision_center_offset.size() == 3) {
      ee_collision_center_offset_ = Vector3d(
          ee_collision_center_offset[0],
          ee_collision_center_offset[1],
          ee_collision_center_offset[2]);
    } else {
      RCLCPP_WARN(
          get_logger(),
          "ee_collision_center_offset must contain 3 values. Using [0, 0, 0].");
    }
    tracking_pos_error_bound_ = std::max(
        0.0,
        declare_parameter<double>("tracking_pos_error_bound", 0.005));

    normal_arrow_length_ = declare_parameter<double>("normal_arrow_length", 0.20);
    arrow_shaft_diameter_ = declare_parameter<double>("arrow_shaft_diameter", 0.01);
    arrow_head_diameter_ = declare_parameter<double>("arrow_head_diameter", 0.02);
    arrow_head_length_ = declare_parameter<double>("arrow_head_length", 0.03);
    const bool configured =
        config_path.empty()
            ? workspace_.configureFromDefaultConfig(get_logger())
            : workspace_.configureFromConfigFile(config_path, get_logger());

    if (!configured) {
      RCLCPP_WARN(get_logger(), "Human workspace markers will not publish until config is valid.");
      config_valid_ = false;
    }

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    marker_pub_ = create_publisher<MarkerArray>(topic_name_, rclcpp::QoS(1).transient_local());

    const auto period = std::chrono::duration<double>(1.0 / publish_rate_hz_);
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        [this]() { publishMarkers(); });

    RCLCPP_INFO(
        get_logger(),
        "Publishing human workspace markers on %s in frame %s.",
        topic_name_.c_str(),
        frame_id_.c_str());
  }

 private:
  void setupMarker(Marker& marker, int id, const std::string& ns, int type) const {
    marker.header.frame_id = frame_id_;
    marker.header.stamp = now();
    marker.ns = ns;
    marker.id = id;
    marker.type = type;
    marker.action = Marker::ADD;
    marker.lifetime = rclcpp::Duration::from_seconds(0.0);
  }

  void publishMarkers() {
    if (!config_valid_) {
      return;
    }

    MarkerArray array;
    const double elapsed_time_sec =
        std::max(0.0, (now() - start_time_).seconds());
    const Vector3d center = workspace_.centerAtTime(elapsed_time_sec);
    const Vector3d direction = workspace_.direction();

    {
      Marker marker;
      setupMarker(marker, 1, "human_workspace_direction", Marker::ARROW);
      marker.points.push_back(toPoint(center));
      marker.points.push_back(toPoint(center + normal_arrow_length_ * direction));
      marker.scale.x = arrow_shaft_diameter_;
      marker.scale.y = arrow_head_diameter_;
      marker.scale.z = arrow_head_length_;
      marker.color = makeColor(0.1f, 0.6f, 1.0f, 0.9f);
      array.markers.push_back(marker);
    }

    {
      Marker marker;
      setupMarker(marker, 2, "human_workspace_motion_sphere", Marker::SPHERE);
      marker.pose.position = toPoint(center);
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 2.0 * workspace_.motionRadius();
      marker.scale.y = 2.0 * workspace_.motionRadius();
      marker.scale.z = 2.0 * workspace_.motionRadius();
      marker.color = makeColor(1.0f, 0.8f, 0.1f, 0.22f);
      array.markers.push_back(marker);
    }

    {
      Marker marker;
      setupMarker(marker, 3, "human_workspace_hand_inflated_sphere", Marker::SPHERE);
      marker.pose.position = toPoint(center);
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 2.0 * workspace_.inflatedHandRadius();
      marker.scale.y = 2.0 * workspace_.inflatedHandRadius();
      marker.scale.z = 2.0 * workspace_.inflatedHandRadius();
      marker.color = makeColor(1.0f, 0.3f, 0.2f, 0.12f);
      array.markers.push_back(marker);
    }

    {
      Marker marker;
      setupMarker(marker, 4, "human_workspace_status", Marker::TEXT_VIEW_FACING);
      marker.action = Marker::DELETE;
      array.markers.push_back(marker);
    }

    publishEndEffectorMarkers(array);

    marker_pub_->publish(array);
  }

  void publishEndEffectorMarkers(MarkerArray& array) {
    if (!visualize_ee_collision_area_) {
      return;
    }

    geometry_msgs::msg::TransformStamped transform;
    try {
      transform = tf_buffer_->lookupTransform(frame_id_, ee_frame_id_, tf2::TimePointZero);
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "Waiting for TF %s -> %s to visualize EE collision area: %s",
          frame_id_.c_str(),
          ee_frame_id_.c_str(),
          ex.what());
      return;
    }

    const Vector3d ee_position(
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z);
    Quaterniond ee_orientation(
        transform.transform.rotation.w,
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z);
    ee_orientation.normalize();
    const Vector3d collision_center =
        ee_position + ee_orientation * ee_collision_center_offset_;
    const double monitored_radius = ee_collision_radius_ + tracking_pos_error_bound_;

    {
      Marker marker;
      setupMarker(marker, 10, "ee_collision_center", Marker::SPHERE);
      marker.pose.position = toPoint(collision_center);
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.03;
      marker.scale.y = 0.03;
      marker.scale.z = 0.03;
      marker.color = makeColor(0.0f, 1.0f, 0.2f, 0.95f);
      array.markers.push_back(marker);
    }

    {
      Marker marker;
      setupMarker(marker, 11, "ee_collision_radius", Marker::SPHERE);
      marker.pose.position = toPoint(collision_center);
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 2.0 * ee_collision_radius_;
      marker.scale.y = 2.0 * ee_collision_radius_;
      marker.scale.z = 2.0 * ee_collision_radius_;
      marker.color = makeColor(0.0f, 1.0f, 0.2f, 0.18f);
      array.markers.push_back(marker);
    }

    {
      Marker marker;
      setupMarker(marker, 12, "ee_monitored_radius", Marker::SPHERE);
      marker.pose.position = toPoint(collision_center);
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 2.0 * monitored_radius;
      marker.scale.y = 2.0 * monitored_radius;
      marker.scale.z = 2.0 * monitored_radius;
      marker.color = makeColor(0.0f, 0.7f, 1.0f, 0.10f);
      array.markers.push_back(marker);
    }

    {
      Marker marker;
      setupMarker(marker, 13, "ee_collision_status", Marker::TEXT_VIEW_FACING);
      marker.action = Marker::DELETE;
      array.markers.push_back(marker);
    }
  }

  HumanWorkspace workspace_;
  bool config_valid_{true};
  std::string frame_id_;
  std::string ee_frame_id_;
  std::string topic_name_;
  bool visualize_ee_collision_area_{true};
  double publish_rate_hz_{10.0};
  double ee_collision_radius_{0.04};
  Vector3d ee_collision_center_offset_{Vector3d::Zero()};
  double tracking_pos_error_bound_{0.005};
  double normal_arrow_length_{0.20};
  double arrow_shaft_diameter_{0.01};
  double arrow_head_diameter_{0.02};
  double arrow_head_length_{0.03};
  rclcpp::Time start_time_;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HumanWorkspaceVisualizer>());
  rclcpp::shutdown();
  return 0;
}
