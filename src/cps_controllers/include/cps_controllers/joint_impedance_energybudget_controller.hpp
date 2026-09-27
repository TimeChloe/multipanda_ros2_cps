// Copyright (c) 2025 Your Organization
//
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <memory>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <realtime_tools/realtime_buffer.hpp>

namespace cps_controllers {

/**
 * @brief Joint-space impedance controller with online energy-budget enforcement
 *        (Lachner et al., IJRR 2021).
 *
 * The controller scales virtual joint stiffness *and* damping online so that
 * the total controlled energy L₍C₎ = T + U never exceeds the user-defined
 * `energy_budget` parameter.
 */
class JointImpedanceEnergyBudgetController : public controller_interface::ControllerInterface {
public:
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
  using Vector7d       = Eigen::Matrix<double, 7, 1>;

  /* ---------- controller_interface hooks ---------- */
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration()   const override;
  controller_interface::return_type           update(const rclcpp::Time& time,
                                                     const rclcpp::Duration& period) override;
  CallbackReturn on_init()                                      override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State&)   override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State&)    override;

private:
  /* ----------------------------- constants ----------------------------- */
  static constexpr int kNumJoints = 7;

  /* ----------------------------- parameters ---------------------------- */
  std::string arm_id_{ "panda" };
  Vector7d    k_nominal_{ Vector7d::Zero() };   ///< user stiffness gains
  Vector7d    d_nominal_{ Vector7d::Zero() };   ///< user damping gains
  double      energy_budget_{ 0.52 };           ///< admissible energy [J]

  /* ------------------ real-time state and goal buffers ----------------- */
  Vector7d q_{ Vector7d::Zero() };
  Vector7d dq_{ Vector7d::Zero() };
  Vector7d dq_filtered_{ Vector7d::Zero() };

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr q_ref_sub_;
  realtime_tools::RealtimeBuffer<Vector7d>                       q_goal_buffer_;

  /* --------------------- Franka robot-model interface ------------------ */
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_model_;

  /* ------------------------------ helpers ------------------------------ */
  void updateJointStates();          ///< reads q_ and dq_ from state_interfaces_

  rclcpp::Time start_time_;          ///< (optional) time marker for trajectories
};

}  // namespace cps_controllers
