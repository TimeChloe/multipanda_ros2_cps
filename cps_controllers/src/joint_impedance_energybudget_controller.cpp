// Copyright (c) 2025 Your Organization
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cps_controllers/joint_impedance_energybudget_controller.hpp"

#include <algorithm>
#include <cmath>

#include <realtime_tools/realtime_buffer.hpp>

namespace cps_controllers {

namespace {
inline double clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }
}  // namespace

/* -------------------------------------------------------------------------- */
/*                                   on_init                                  */
/* -------------------------------------------------------------------------- */

JointImpedanceEnergyBudgetController::CallbackReturn
JointImpedanceEnergyBudgetController::on_init() {
  try {
    auto_declare<std::string>("arm_id", arm_id_);
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<double>("energy_budget", energy_budget_);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Init exception: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

/* -------------------------------------------------------------------------- */
/*                                 on_configure                               */
/* -------------------------------------------------------------------------- */

JointImpedanceEnergyBudgetController::CallbackReturn
JointImpedanceEnergyBudgetController::on_configure(const rclcpp_lifecycle::State&) {
  arm_id_        = get_node()->get_parameter("arm_id").as_string();
  energy_budget_ = get_node()->get_parameter("energy_budget").as_double();

  auto k_g = get_node()->get_parameter("k_gains").as_double_array();
  auto d_g = get_node()->get_parameter("d_gains").as_double_array();
  if (k_g.size() != kNumJoints || d_g.size() != kNumJoints) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "k_gains and d_gains must each contain %d elements", kNumJoints);
    return CallbackReturn::FAILURE;
  }
  for (int i = 0; i < kNumJoints; ++i) {
    k_nominal_(i) = k_g[i];
    d_nominal_(i) = d_g[i];
  }

  franka_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      arm_id_ + "/robot_model", arm_id_);

  q_ref_sub_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      "desired_q", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (msg->position.size() == kNumJoints) {
          Vector7d q_des;
          Eigen::VectorXd::Map(&q_des[0], kNumJoints) =
              Eigen::VectorXd::Map(msg->position.data(), kNumJoints);
          q_goal_buffer_.writeFromNonRT(q_des);
        }
      });

  return CallbackReturn::SUCCESS;
}

/* -------------------------------------------------------------------------- */
/*                                 on_activate                                */
/* -------------------------------------------------------------------------- */

JointImpedanceEnergyBudgetController::CallbackReturn
JointImpedanceEnergyBudgetController::on_activate(const rclcpp_lifecycle::State&) {
  updateJointStates();
  franka_model_->assign_loaned_state_interfaces(state_interfaces_);
  q_goal_buffer_.writeFromNonRT(q_);
  start_time_ = get_node()->now();
  return CallbackReturn::SUCCESS;
}

/* -------------------------------------------------------------------------- */
/*                       command / state interface setups                      */
/* -------------------------------------------------------------------------- */

controller_interface::InterfaceConfiguration
JointImpedanceEnergyBudgetController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration cfg;
  cfg.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    cfg.names.emplace_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return cfg;
}

controller_interface::InterfaceConfiguration
JointImpedanceEnergyBudgetController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration cfg;
  cfg.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    cfg.names.emplace_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    cfg.names.emplace_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  for (const auto& n : franka_model_->get_state_interface_names()) {
    cfg.names.push_back(n);
  }
  return cfg;
}

/* -------------------------------------------------------------------------- */
/*                                   update                                   */
/* -------------------------------------------------------------------------- */

controller_interface::return_type JointImpedanceEnergyBudgetController::update(
    const rclcpp::Time&, const rclcpp::Duration&) {
  updateJointStates();

  const Vector7d q_goal = *q_goal_buffer_.readFromRT();
  const Vector7d q_err  = q_goal - q_;

  const double potential_energy = 0.5 * (k_nominal_.cwiseProduct(q_err.cwiseAbs2())).sum();

  const auto& mass_vec = franka_model_->getMassMatrix();
  Eigen::Map<const Eigen::Matrix<double, 7, 7>> M(mass_vec.data());
  const double kinetic_energy = 0.5 * dq_.transpose() * M * dq_;

  double k_scale = 1.0;
  if (potential_energy > 1e-9) {
    const double total_E = potential_energy + kinetic_energy;
    if (total_E > energy_budget_) {
      k_scale = clamp01((energy_budget_ - kinetic_energy) / potential_energy);
    }
  } else {
    k_scale = 0.0;
  }

  const Vector7d k_scaled = k_scale * k_nominal_;
  const Vector7d d_scaled = std::sqrt(k_scale) * d_nominal_;

  constexpr double alpha = 0.99;
  dq_filtered_ = (1.0 - alpha) * dq_filtered_ + alpha * dq_;

  Eigen::Map<const Vector7d> coriolis(franka_model_->getCoriolisForceVector().data());
  const Vector7d tau_cmd = k_scaled.cwiseProduct(q_err) +
                           d_scaled.cwiseProduct(-dq_filtered_) + coriolis;

  for (int i = 0; i < kNumJoints; ++i) {
    command_interfaces_[i].set_value(tau_cmd(i));
  }

  return controller_interface::return_type::OK;
}

/* -------------------------------------------------------------------------- */
/*                              updateJointStates                             */
/* -------------------------------------------------------------------------- */

void JointImpedanceEnergyBudgetController::updateJointStates() {
  for (int i = 0; i < kNumJoints; ++i) {
    q_(i)  = state_interfaces_[2 * i].get_value();
    dq_(i) = state_interfaces_[2 * i + 1].get_value();
  }
}

}  // namespace cps_controllers

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(cps_controllers::JointImpedanceEnergyBudgetController,
                       controller_interface::ControllerInterface)
