# Table Corner Approach Via Point Scenarios

These scenarios are intentionally mild. Each path first moves to an upper
corner around the current TCP pose, then approaches the table/contact area with
a smooth inward descent.

All examples use:

- `cartesian_via_points_relative: true`
- `cartesian_via_point_quaternions_relative: true`
- the current TCP pose as the coordinate origin
- identity quaternions `[0, 0, 0, 1]`, which keep the initial TCP orientation
- a final approach point near `[0.00, 0.00, -0.42]`

## Scenarios

- `reachable_via_points_table_corner_front_left.yaml`
- `reachable_via_points_table_corner_front_right.yaml`
- `reachable_via_points_table_corner_back_left.yaml`
- `reachable_via_points_table_corner_back_right.yaml`
- `reachable_via_points_table_center_high.yaml`

Copy one scenario's `cartesian_via_points`,
`cartesian_via_point_quaternions_relative`, and
`cartesian_via_point_quaternions` blocks into the active controller YAML, or pass
them as a ROS parameter overlay if your launch setup supports that. Quaternion
entries use `[x, y, z, w]`.
