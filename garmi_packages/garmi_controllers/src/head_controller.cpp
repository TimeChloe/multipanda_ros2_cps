#include "garmi_controllers/head_controller.hpp"
// #include "../include/garmi_controllers/head_controller.hpp"
#include <rclcpp/duration.hpp>

using std::placeholders::_1;

namespace garmi_controllers{


controller_interface::InterfaceConfiguration 
  HeadController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.push_back(robot_id_ + "_joint1/effort");
  config.names.push_back(robot_id_ + "_joint2/effort");
  return config;
};

controller_interface::InterfaceConfiguration 
  HeadController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.push_back(robot_id_ + "_joint1/position");
  config.names.push_back(robot_id_ + "_joint1/velocity");
  config.names.push_back(robot_id_ + "_joint1/effort");
  config.names.push_back(robot_id_ + "_joint2/position");
  config.names.push_back(robot_id_ + "_joint2/velocity");
  config.names.push_back(robot_id_ + "_joint2/effort");
  return config;
};

controller_interface::return_type HeadController::update(const rclcpp::Time& time,
                                                               const rclcpp::Duration& period){
  current_time_ = this->get_node()->now();
  // RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000, "\nLeft vel:  %f\nRight vel: %f", state_interfaces_.at(0).get_value(), state_interfaces_.at(2).get_value());
  
  head_state_.set_theta(0, state_interfaces_.at(0).get_value()); // left-right
  head_state_.set_theta(1, state_interfaces_.at(3).get_value()); // down-up

  // PD control 
  double time_diff = 0;
  for (size_t joint_number = 0; joint_number < 2; joint_number++) {
    
    // Time difference between two commands
    time_diff = (this->get_node()->now() - last_time_).seconds();
    
    // Get current and previous theta values
    double theta = head_state_.get_theta(joint_number);
    double theta_previous = head_state_.get_theta_previous(joint_number);

    // Obtain velocities
    double dtheta_d = 0;
    double dtheta = (theta - theta_previous) / time_diff;
    head_state_.set_dtheta(joint_number, dtheta);

    theta_d_[joint_number] = theta_d_[joint_number] * (1.0 - pt1_filter_) + pt1_filter_ * head_state_.get_theta_target(
                         joint_number);

    // Calculate the approapriate torque value - PD control    
    double tau_j = std::min(15.0, std::max(-15.0, (theta_d_[joint_number] - theta) * k_p_ + 
                                          (dtheta_d - dtheta) * k_d_));
    
    // Set the values to the head_state_
    head_state_.set_dtheta(joint_number, dtheta);
    head_state_.set_theta(joint_number, theta);
    head_state_.set_tau_j(joint_number, tau_j);
 
    head_state_.set_theta_previous(joint_number, theta);
  }

  // Command the calculated torque values
  command_interfaces_[0].set_value(head_state_.get_tau_j(0));
  command_interfaces_[1].set_value(head_state_.get_tau_j(1));

  last_time_ = current_time_;
  return controller_interface::return_type::OK;
  };


  

CallbackReturn HeadController::on_init(){
  // nothing for now
  try {
    auto_declare<std::string>("robot_id", "garmi_head");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
} ;

CallbackReturn HeadController::on_configure(const rclcpp_lifecycle::State& previous_state){
  // nothing for now
  robot_id_ = get_node()->get_parameter("robot_id").as_string();
  is_sim_ = get_node()->get_parameter("sim").as_bool();
  goal_pos_sub_ = this->get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(robot_id_ + "/head_command", 1, std::bind(&HeadController::command_listener, this, _1));
  return CallbackReturn::SUCCESS;
} ;

CallbackReturn HeadController::on_activate(const rclcpp_lifecycle::State& previous_state){
  current_time_ = this->get_node()->now();
  last_time_ = this->get_node()->now();
  return CallbackReturn::SUCCESS;
} ;

CallbackReturn HeadController::on_deactivate(const rclcpp_lifecycle::State& previous_state){
  return CallbackReturn::SUCCESS;
} ;
}   // namespace
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(garmi_controllers::HeadController,
                       controller_interface::ControllerInterface)