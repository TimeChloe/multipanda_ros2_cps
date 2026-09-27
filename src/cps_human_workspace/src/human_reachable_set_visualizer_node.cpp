#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "cps_human_workspace/msg/human_reachable_set.hpp"

namespace
{

constexpr char kMarkerNamespace[] = "sara_human_hand_reachable_set";

class HumanReachableSetVisualizer : public rclcpp::Node
{
public:
  HumanReachableSetVisualizer()
  : Node("human_reachable_set_visualizer")
  {
    const std::string reachable_set_topic = declare_parameter<std::string>(
      "reachable_set_topic", "/human_workspace/reachable_set");
    const std::string marker_topic = declare_parameter<std::string>(
      "marker_topic", "/human_workspace/markers");
    marker_alpha_ = std::clamp(
      declare_parameter<double>("marker_alpha", 0.3), 0.01, 1.0);
    marker_lifetime_sec_ = std::max(
      0.0, declare_parameter<double>("marker_lifetime_sec", 1.0));

    if (reachable_set_topic.empty() || marker_topic.empty()) {
      throw std::invalid_argument(
              "reachable_set_topic and marker_topic must not be empty");
    }

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      marker_topic, rclcpp::QoS(1).reliable().transient_local());
    reachable_set_sub_ =
      create_subscription<cps_human_workspace::msg::HumanReachableSet>(
      reachable_set_topic, rclcpp::QoS(1).reliable().transient_local(),
      [this](
        const cps_human_workspace::msg::HumanReachableSet::SharedPtr message) {
        publishMarker(*message);
      });

    RCLCPP_INFO(
      get_logger(),
      "Visualizing calculated human reachable sets from '%s' on '%s'.",
      reachable_set_topic.c_str(), marker_topic.c_str());
  }

private:
  void publishMarker(
    const cps_human_workspace::msg::HumanReachableSet & reachable_set)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = reachable_set.header;
    marker.ns = kMarkerNamespace;
    marker.id = 0;

    const bool finite_geometry =
      std::isfinite(reachable_set.center.x) &&
      std::isfinite(reachable_set.center.y) &&
      std::isfinite(reachable_set.center.z) &&
      std::isfinite(reachable_set.radius) && reachable_set.radius >= 0.0;
    if (reachable_set.valid && finite_geometry &&
      !reachable_set.header.frame_id.empty())
    {
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position = reachable_set.center;
      marker.pose.orientation.w = 1.0;
      const double diameter = 2.0 * reachable_set.radius;
      marker.scale.x = diameter;
      marker.scale.y = diameter;
      marker.scale.z = diameter;
      marker.color.r = 0.0F;
      marker.color.g = 0.0F;
      marker.color.b = 1.0F;
      marker.color.a = static_cast<float>(marker_alpha_);
      marker.lifetime =
        rclcpp::Duration::from_seconds(marker_lifetime_sec_);
    } else {
      marker.action = visualization_msgs::msg::Marker::DELETE;
    }

    visualization_msgs::msg::MarkerArray marker_array;
    marker_array.markers.push_back(std::move(marker));
    marker_pub_->publish(marker_array);
  }

  double marker_alpha_{0.3};
  double marker_lifetime_sec_{1.0};
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Subscription<
    cps_human_workspace::msg::HumanReachableSet>::SharedPtr reachable_set_sub_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HumanReachableSetVisualizer>());
  rclcpp::shutdown();
  return 0;
}
