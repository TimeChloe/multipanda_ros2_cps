#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>

#include "cps_human_workspace/human_workspace.hpp"
#include "cps_human_workspace/human_workspace_message.hpp"
#include "cps_human_workspace/msg/human_workspace.hpp"

namespace
{

using cps_human_workspace::HumanWorkspace;
using HumanWorkspaceMsg = cps_human_workspace::msg::HumanWorkspace;

class HumanWorkspacePublisher : public rclcpp::Node
{
public:
  HumanWorkspacePublisher()
  : Node("human_workspace_publisher")
  {
    start_time_ = now();
    const std::string default_config_path =
      ament_index_cpp::get_package_share_directory("cps_human_workspace") +
      "/config/human_workspace.yaml";
    const std::string requested_config_path =
      declare_parameter<std::string>(
      "human_workspace_config_path", default_config_path);
    const std::string config_path = requested_config_path.empty() ?
      default_config_path : requested_config_path;
    frame_id_ = declare_parameter<std::string>("frame_id", "panda_link0");
    const std::string state_topic_name =
      declare_parameter<std::string>("state_topic", "human_workspace/state");
    const double publish_rate_hz =
      declare_parameter<double>("publish_rate", 10.0);

    if (frame_id_.empty() || state_topic_name.empty()) {
      throw std::invalid_argument("frame_id and state_topic must not be empty");
    }
    if (!std::isfinite(publish_rate_hz) || publish_rate_hz <= 0.0) {
      throw std::invalid_argument("publish_rate must be finite and positive");
    }
    if (!workspace_.configureFromConfigFile(config_path, get_logger())) {
      throw std::runtime_error("could not configure human workspace publisher");
    }

    state_pub_ = create_publisher<HumanWorkspaceMsg>(
      state_topic_name,
      rclcpp::QoS(1).transient_local());

    const auto period = std::chrono::duration<double>(1.0 / publish_rate_hz);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() {publishObservation();});

    RCLCPP_INFO(
      get_logger(),
      "Publishing timestamped human observations on %s in frame %s.",
      state_topic_name.c_str(),
      frame_id_.c_str());
  }

private:
  void publishObservation()
  {
    const double elapsed_time_sec =
      std::max(0.0, (now() - start_time_).seconds());
    const auto center = workspace_.centerAtTime(elapsed_time_sec);
    const auto velocity =
      workspace_.centerVelocityAtTime(elapsed_time_sec);

    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = frame_id_;
    state_pub_->publish(
      cps_human_workspace::makeHumanWorkspaceMessage(
        header, center, velocity, workspace_.parameters()));
  }

  HumanWorkspace workspace_;
  std::string frame_id_;
  rclcpp::Time start_time_;
  rclcpp::Publisher<HumanWorkspaceMsg>::SharedPtr state_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HumanWorkspacePublisher>());
  rclcpp::shutdown();
  return 0;
}
