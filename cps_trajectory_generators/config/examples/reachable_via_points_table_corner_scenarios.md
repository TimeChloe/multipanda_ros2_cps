# Table Corner Approach Via Point Scenarios

These scenarios are intentionally mild. Each path first moves to an upper
corner in the robot base frame, then approaches the table/contact area with a
smooth inward descent.

All examples use:

- robot base frame (`panda_link0`) coordinates
- one 7-value state per via point: `[x, y, z, qx, qy, qz, qw]`
- the default sim activation orientation
  `[0.923956, -0.382499, 0.000000, 0.000000]`
- a final approach point near `[0.306957, 0.000000, 0.279912]`

## Scenarios

- `reachable_via_points_table_corner_front_left.yaml`
- `reachable_via_points_table_corner_front_right.yaml`
- `reachable_via_points_table_corner_back_left.yaml`
- `reachable_via_points_table_corner_back_right.yaml`
- `reachable_via_points_table_center_high.yaml`

Copy one scenario's `cartesian_via_points` block into the active controller
YAML, or pass it as a ROS parameter overlay if your launch setup supports that.
Quaternion entries use `[x, y, z, w]`.
