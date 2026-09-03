std_msgs/Header header

geometry_msgs/Point sphere_center

geometry_msgs/Vector3 center_velocity

# Physical radius of the measured hand enclosure. Reachability uncertainty is
# generated separately from the fields below.
float64 motion_radius

# SaRA ReachLib BodyPartCombined single-hand model. The published
# sphere_center/center_velocity form the timestamped current observation.
float64 hand_max_velocity
float64 hand_max_acceleration
float64 measurement_error_position
float64 measurement_error_velocity
float64 measurement_delay
