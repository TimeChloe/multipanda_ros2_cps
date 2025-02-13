// Copyright (c) 2021 Franka Emika GmbH
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

#pragma once

#include <string>

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/std_msgs/msg/float64_multi_array.hpp>


#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using nav_msgs::msg::Odometry;
using sensor_msgs::msg::JointState;
using std_msgs::msg::Float64MultiArray;


namespace garmi_controllers {

class State {
  public:
    double theta_[2] = {0.0, 0.0};
    double theta_previous_[2] = {0.0, 0.0};

    double dtheta_[2] = {0.0, 0.0};
    double theta_target_[2] = {0.0, 0.0};
    double tau_j_[2] = {0.0, 0.0};
    double tau_j_d_[2] = {0.0, 0.0};
    std::mutex mux_;


    void set_theta(int index, double theta) {
        mux_.lock();
        theta_[index] = theta;
        mux_.unlock();
    }

    double get_theta(int index) {
        mux_.lock();
        double theta = theta_[index];
        mux_.unlock();
        return theta;
    }

     void set_theta_previous(int index, double theta_previous) {
        mux_.lock();
        theta_previous_[index] = theta_previous;
        mux_.unlock();
    }

    double get_theta_previous(int index) {
        mux_.lock();
        double theta_previous = theta_previous_[index];
        mux_.unlock();
        return theta_previous;
    }


    void set_dtheta(int index, double dtheta) {
        mux_.lock();
        dtheta_[index] = dtheta;
        mux_.unlock();
    }

    double get_dtheta(int index) {
        mux_.lock();
        double dtheta = dtheta_[index];
        mux_.unlock();
        return dtheta;
    }

    void set_theta_target(double theta_target_1, double theta_target_2) {
        mux_.lock();
        theta_target_[0] = theta_target_1;
        theta_target_[1] = theta_target_2;
        mux_.unlock();
    }

    double get_theta_target(int index) {
        mux_.lock();
        double theta_goal = theta_target_[index];
        mux_.unlock();
        return theta_goal;
    }

    void set_tau_j(int index, double tau_j) {
        mux_.lock();
        tau_j_[index] = tau_j;
        mux_.unlock();
    }

    double get_tau_j(int index) {
        mux_.lock();
        double tau_j = tau_j_[index];
        mux_.unlock();
        return tau_j;
    }

    void set_tau_j_d(int index, double tau_j_d) {
        mux_.lock();
        tau_j_d_[index] = tau_j_d;
        mux_.unlock();
    }

    double get_tau_j_d(int index) {
        mux_.lock();
        double tau_j_d = tau_j_d_[index];
        mux_.unlock();
        return tau_j_d;
    }
};


class HeadController : public controller_interface::ControllerInterface {
  public:
    controller_interface::InterfaceConfiguration command_interface_configuration() const override;
    controller_interface::InterfaceConfiguration state_interface_configuration() const override;
    controller_interface::return_type update(const rclcpp::Time& time,
                                            const rclcpp::Duration& period) override;
    CallbackReturn on_init() override;
    CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
    CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  private:
  
    // basic parameters
    std::string robot_id_ = "garmi_head";
    bool is_sim_ = true;
    const int num_joints = 2;

    // head parameters
    double k_p_ = 200.0;
    double k_p_target_ = 200.0;
    double k_d_ = 5.0;
    double k_d_target_ = 5.0;
    double pt1_filter_ = 0.001;
    double pt1_filter_target_ = 0;
    std::vector<double> theta_d_{0.0,0.0};

    std::shared_ptr<rclcpp::Subscription<Float64MultiArray>> goal_pos_sub_;
    std::shared_ptr<rclcpp::Publisher<JointState>> joint_state_pub_;
    std::shared_ptr<rclcpp::Publisher<Odometry>> head_odom_pub_;

    State head_state_;

    rclcpp::Time current_time_, last_time_, last_command_time_;
    rclcpp::Clock clock_;

    
    void command_listener(const Float64MultiArray pan_tilt) {
      head_state_.set_theta_target(std::min(0.6, std::max(-0.6, pan_tilt.data[0])),
                                  std::min(0.2, std::max(-0.2, pan_tilt.data[1])));
    }
  };

}  // namespace franka_example_controllers